import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 导入你的自定义模块
from dataset.io import load_dataset
from dataset.preprocess import prepare_batches, NORM_SCALE, D_MAX, SEQ_LEN
from model.wgan import Generator, Discriminator

# --- 保持原有常量定义 ---
LATENT_DIM = 64
LAMBDA_GP = 10.0
LAMBDA_GEOM = 2.0
LAMBDA_SMOOTH = 2.0
LAMBDA_SPREAD = 3.0
LAMBDA_JITTER = 0.2
LAMBDA_ACC = 1.0
MIN_ACC_STD = 0.03
JITTER_WINDOW = 8
LAMBDA_DRAWDOWN = 2.0
DRAWDOWN_MARGIN_PX = 2.0
D_STEPS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# 辅助函数：梯度惩罚 (WGAN-GP)
# ============================
def gradient_penalty(D, real, fake, cond):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, device=real.device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = D(interp, cond)
    grad = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad = grad.reshape(batch_size, -1)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()

# ============================
# 优化后的轨迹 Loss 类
# ============================
class OptimizedTrajectoryLoss:
    def __init__(self, norm_scale, d_max):
        self.norm_scale = float(norm_scale)
        self.d_max = float(d_max)

    def get_losses(self, fake_seq, cond, args):
        """
        fake_seq: (B, L, 2) -> (dx, dy)
        cond: (B, 1) -> target_dist / D_MAX
        """
        dx = fake_seq[:, :, 0]
        dy = fake_seq[:, :, 1]
        x_cum = torch.cumsum(dx, dim=1)
        y_cum = torch.cumsum(dy, dim=1)
        
        vel = fake_seq
        accel = torch.diff(vel, dim=1)
        jerk = torch.diff(accel, dim=1)

        # 1. 几何与对齐：X轴到终点，Y轴回原点
        target_val = cond.squeeze(1) * (self.d_max / self.norm_scale)
        loss_geom_x = F.mse_loss(x_cum[:, -1], target_val)
        loss_geom_y = torch.mean(y_cum[:, -1]**2)
        
        # 2. 物理平滑度：惩罚三阶导(jerk)和Y轴剧烈加速度
        loss_smooth = torch.mean(jerk**2)
        loss_accel_y = torch.mean(accel[:, :, 1]**2)

        # 3. Y轴边界：限制上下漂移范围
        y_range = torch.max(y_cum, dim=1).values - torch.min(y_cum, dim=1).values
        margin_y = 5.0 / self.norm_scale
        loss_y_range = torch.mean(F.relu(y_range - margin_y)**2)

        # 4. 回撤损失
        run_max = torch.cummax(x_cum, dim=1).values
        dd = run_max - x_cum
        margin_dd = args.drawdown_margin_px / self.norm_scale
        loss_drawdown = torch.mean(F.relu(dd - margin_dd)**2)

        # 5. 分散度损失：单步最大占比不宜过高
        max_step_ratio = (dx.abs() / (dx.abs().sum(dim=1, keepdim=True) + 1e-6)).max(dim=1)[0]
        loss_spread = torch.mean(F.relu(max_step_ratio - 0.15))

        # 6. 终点稳定性：最后10%步长模拟减速对准
        end_vel = vel[:, -int(SEQ_LEN*0.1):, :]
        loss_stop = torch.mean(end_vel**2)

        # 7. 抖动约束：与可视化一致，用 ds=sqrt(dx^2+dy^2) 的滑动窗口方差
        #    人类轨迹 jitter (disp. var.) 约 100~250（像素²），归一化约 0.5~2.5
        ds = torch.sqrt(dx**2 + dy**2 + 1e-12)
        if ds.size(1) >= args.jitter_window:
            var_local = ds.unfold(1, args.jitter_window, 1).var(dim=-1)
            target_var = getattr(args, "jitter_target_var", 0.5)
            loss_jitter = (target_var - var_local).clamp(min=0).mean()
        else:
            loss_jitter = torch.tensor(0.0, device=dx.device)

        total_aux_loss = (
            args.lambda_geom * (loss_geom_x + 0.5 * loss_geom_y) +
            args.lambda_smooth * loss_smooth +
            args.lambda_acc * loss_accel_y +
            args.lambda_drawdown * loss_drawdown +
            args.lambda_spread * loss_spread +
            args.lambda_jitter * loss_jitter +
            1.0 * loss_y_range + 
            2.0 * loss_stop
        )
        
        return total_aux_loss, {
            "geom": loss_geom_x.item(),
            "smooth": loss_smooth.item(),
            "dd": loss_drawdown.item(),
            "stop": loss_stop.item(),
            "jitter": loss_jitter.item()
        }


def _plot_training_curves(history, out_dir):
    """绘制训练过程中的 D/G 损失与辅助损失曲线，保存到 out_dir/training_curves.png。"""
    epochs = range(1, len(history["d_loss"]) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 上图：判别器与生成器损失
    ax0 = axes[0]
    ax0.plot(epochs, history["d_loss"], label="D loss", color="C0", alpha=0.9)
    ax0.plot(epochs, history["g_loss"], label="G loss", color="C1", alpha=0.9)
    ax0.set_ylabel("Loss")
    ax0.set_title("WGAN-GP: Discriminator & Generator Loss")
    ax0.legend(loc="upper right")
    ax0.grid(True, alpha=0.3)

    # 下图：辅助损失分量（几何、平滑、回撤、终点、抖动）
    ax1 = axes[1]
    ax1.plot(epochs, history["geom"], label="geom", color="C2", alpha=0.8)
    ax1.plot(epochs, history["smooth"], label="smooth", color="C3", alpha=0.8)
    ax1.plot(epochs, history["dd"], label="drawdown", color="C4", alpha=0.8)
    ax1.plot(epochs, history["stop"], label="stop", color="C5", alpha=0.8)
    ax1.plot(epochs, history["jitter"], label="jitter", color="C6", alpha=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Aux Loss")
    ax1.set_title("Auxiliary Loss Components (geom, smooth, drawdown, stop, jitter)")
    ax1.legend(loc="upper right", ncol=2)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curves saved to {out_path}")


# ============================
# 训练主循环
# ============================
def train():
    parser = argparse.ArgumentParser(description="Train WGAN-GP (v2) Optimized")
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=str, default="checkpoints")
    parser.add_argument("--d-steps", type=int, default=D_STEPS)
    parser.add_argument("--lambda-gp", type=float, default=LAMBDA_GP)
    parser.add_argument("--lambda-geom", type=float, default=LAMBDA_GEOM)
    parser.add_argument("--lambda-smooth", type=float, default=LAMBDA_SMOOTH)
    parser.add_argument("--lambda-spread", type=float, default=LAMBDA_SPREAD)
    parser.add_argument("--lambda-jitter", type=float, default=LAMBDA_JITTER)
    parser.add_argument("--lambda-acc", type=float, default=LAMBDA_ACC)
    parser.add_argument("--jitter-window", type=int, default=JITTER_WINDOW)
    parser.add_argument("--jitter-target-var", type=float, default=0.5,
                        help="目标滑动方差（归一化空间），约 0.5 对应像素²约 50")
    parser.add_argument("--min-acc-std", type=float, default=MIN_ACC_STD)
    parser.add_argument("--lambda-drawdown", type=float, default=LAMBDA_DRAWDOWN)
    parser.add_argument("--drawdown-margin-px", type=float, default=DRAWDOWN_MARGIN_PX)
    args = parser.parse_args()

    # ---- 数据加载 ----
    train_dir = Path(args.data_dir) / "train"
    train_samples = load_dataset(train_dir)
    if not train_samples:
        print("Error: No data found.")
        return

    # ---- 模型与优化器 ----
    G = Generator(max_len=SEQ_LEN).to(DEVICE)
    D = Discriminator(max_len=SEQ_LEN).to(DEVICE)
    g_opt = Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    d_opt = Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.9))
    
    criterion = OptimizedTrajectoryLoss(NORM_SCALE, D_MAX)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # 训练过程记录，用于绘图
    history = {
        "d_loss": [],
        "g_loss": [],
        "geom": [],
        "smooth": [],
        "dd": [],
        "stop": [],
        "jitter": [],
    }

    print(f"Starting training on {DEVICE}...")

    for epoch in range(args.epochs):
        G.train()
        D.train()

        d_loss_log, g_loss_log = 0.0, 0.0
        n_d, n_g = 0, 0
        aux_accum = {"geom": 0.0, "smooth": 0.0, "dd": 0.0, "stop": 0.0, "jitter": 0.0}
        n_aux = 0
        aux_metrics = {}

        # prepare_batches 返回 (cond, seq) 的迭代器
        for step, (cond_np, seq_np) in enumerate(
            prepare_batches(train_samples, max_len=SEQ_LEN, batch_size=args.batch_size, shuffle=True)
        ):
            cond = torch.from_numpy(cond_np).float().to(DEVICE)
            real = torch.from_numpy(seq_np).float().to(DEVICE)
            B = real.size(0)

            # ---- 1. 训练判别器 D ----
            z = torch.randn(B, LATENT_DIM, device=DEVICE)
            fake = G(z, cond).detach()
            
            d_real = D(real, cond)
            d_fake = D(fake, cond)
            gp = gradient_penalty(D, real, fake, cond)
            
            # WGAN 损失：E[D(fake)] - E[D(real)]
            d_loss = d_fake.mean() - d_real.mean() + args.lambda_gp * gp

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()
            
            d_loss_log += d_loss.item()
            n_d += 1

            # ---- 2. 训练生成器 G (按比例执行) ----
            if step % args.d_steps == 0:
                z = torch.randn(B, LATENT_DIM, device=DEVICE)
                gen_seq = G(z, cond)

                # 对抗损失
                loss_adv = -D(gen_seq, cond).mean()
                
                # 物理与几何辅助损失
                loss_aux, aux_metrics = criterion.get_losses(gen_seq, cond, args)
                
                # 多样性损失：惩罚 Batch 内轨迹完全一致
                dist_matrix = torch.cdist(gen_seq.view(B, -1), gen_seq.view(B, -1))
                loss_diversity = -torch.log(dist_matrix.mean() + 1e-6)

                total_g_loss = loss_adv + loss_aux + 0.05 * loss_diversity

                g_opt.zero_grad()
                total_g_loss.backward()
                
                # 关键：梯度裁剪，防止 LSTM 训练崩溃
                torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
                
                g_opt.step()

                g_loss_log += total_g_loss.item()
                n_g += 1
                for k in aux_accum:
                    aux_accum[k] += aux_metrics.get(k, 0.0)
                n_aux += 1

        # ---- 记录本 epoch 用于绘图 ----
        history["d_loss"].append(d_loss_log / n_d)
        history["g_loss"].append(g_loss_log / max(n_g, 1))
        if n_aux > 0:
            for k in aux_accum:
                history[k].append(aux_accum[k] / n_aux)
        else:
            for k in aux_accum:
                history[k].append(0.0)

        # ---- 日志打印 ----
        if (epoch + 1) % 100 == 0 or epoch == 0:
            avg_d = d_loss_log / n_d
            avg_g = g_loss_log / max(n_g, 1)
            print(f"Epoch [{epoch+1}/{args.epochs}] | D: {avg_d:.4f} | G: {avg_g:.4f}")
            # 每 500 epoch 打印一次辅助损失分量，便于调参
            if (epoch + 1) % 500 == 0 and n_g > 0 and aux_metrics:
                print("   aux: geom=%.4f smooth=%.4f dd=%.4f stop=%.4f jitter=%.4f" % (
                    aux_metrics.get("geom", 0), aux_metrics.get("smooth", 0),
                    aux_metrics.get("dd", 0), aux_metrics.get("stop", 0), aux_metrics.get("jitter", 0)))

        # ---- 定期保存 ----
        if (epoch + 1) % 500 == 0:
            ckpt_path = Path(args.out) / f"wgan_v2_epoch{epoch+1}.pt"
            torch.save({"G": G.state_dict(), "D": D.state_dict()}, ckpt_path)
            # 定期绘制当前曲线，便于长时间训练时查看
            _plot_training_curves(history, Path(args.out))

    # ---- 绘制训练曲线并保存 ----
    _plot_training_curves(history, Path(args.out))

    # ---- 最终保存（兼容推理/README 默认使用的 wgan.pt）----
    final_path = Path(args.out) / "wgan_v2_final.pt"
    torch.save({"G": G.state_dict(), "D": D.state_dict()}, final_path)
    compat_path = Path(args.out) / "wgan.pt"
    if compat_path.resolve() != final_path.resolve():
        import shutil
        shutil.copy(final_path, compat_path)
        print(f"已另存为 {compat_path} 供推理/对比使用。")
    print("Training Complete.")

if __name__ == "__main__":
    train()