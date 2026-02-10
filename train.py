"""
训练脚本（v2 思路）：
- 重采样预处理（等时间间隔插值，固定序列长度 128）
- 2D 输出 (dx, dy)，去掉 dt
- WGAN-GP：D 训练 D_STEPS 次，G 训练 1 次
- 辅助 Loss：geometry（终点）+ smoothness（jerk）+ spread（防集中）
"""
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from dataset.io import load_dataset
from dataset.preprocess import prepare_batches, NORM_SCALE, D_MAX, SEQ_LEN
from model.wgan import Generator, Discriminator

LATENT_DIM = 64
LAMBDA_GP = 10.0        # WGAN-GP 梯度惩罚
LAMBDA_GEOM = 2.0       # 几何约束（终点距离），降低避免主导
LAMBDA_SMOOTH = 2.0     # 平滑约束（惩罚 jerk），过大会压制抖动
LAMBDA_SPREAD = 3.0     # 分散约束：防止位移集中在少数步
LAMBDA_JITTER = 0.2     # 最小抖动约束：鼓励 dx 有人手般的自然波动（过大会引入过强波动）
LAMBDA_ACC = 1.0        # 加速度强度约束：鼓励速度有阶跃变化，避免加速度接近 0
MIN_ACC_STD = 0.03      # 归一化空间中「速度差分」的最小标准差（对应加速度有起伏）
JITTER_WINDOW = 8       # 0=仅全局抖动；8 等=按窗口约束局部抖动，更易产生加速度起伏
LAMBDA_DRAWDOWN = 2.0   # 回撤约束：惩罚 x(t) 从历史高点的大幅回撤
DRAWDOWN_MARGIN_PX = 2.0  # 允许的微小回撤（px），超过部分才惩罚
D_STEPS = 3             # D/G 训练比（减少以获得更多 G 更新）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================
# 辅助 Loss
# ============================
def gradient_penalty(D, real, fake, cond):
    """WGAN-GP 梯度惩罚。"""
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


def geometry_loss(fake_seq, cond):
    """
    几何约束：生成轨迹的 sum(dx) 应等于目标距离。
    fake_seq: (B, L, 2), 归一化空间
    cond: (B, 1), target_dist / D_MAX
    """
    total_dx = fake_seq[:, :, 0].sum(dim=1)
    target_val = cond.squeeze(1) * (D_MAX / NORM_SCALE)
    return F.mse_loss(total_dx, target_val)


def smoothness_loss(fake_seq):
    """
    平滑约束：惩罚 jerk（三阶运动量），消除生成轨迹的尖刺。
    fake_seq: (B, L, 2)
    """
    vel_diff = fake_seq[:, 1:, :] - fake_seq[:, :-1, :]        # 一阶差分
    acc_diff = vel_diff[:, 1:, :] - vel_diff[:, :-1, :]         # 二阶差分 (jerk)
    return torch.mean(acc_diff ** 2)


def spread_loss(fake_seq):
    """
    分散约束：防止位移集中在少数几步。
    - 惩罚单步最大贡献超过阈值（对 128 步，理想 < 5%）
    - 惩罚负向位移（滑块应基本向前）
    fake_seq: (B, L, 2)
    """
    dx = fake_seq[:, :, 0]                                      # (B, L)
    abs_dx = dx.abs()
    total = abs_dx.sum(dim=1, keepdim=True).clamp(min=1e-6)
    # 最大单步占比
    max_contrib = (abs_dx / total).max(dim=1)[0]                # (B,)
    concentration = (max_contrib - 0.05).clamp(min=0).mean()
    # 负向位移惩罚（滑块应向前移动）
    backward = (-dx).clamp(min=0).mean()
    return concentration + backward


def jitter_loss(fake_seq, min_std=0.06, window=0):
    """
    最小抖动约束：鼓励 dx 沿时间有自然波动，避免过于平滑。
    window=0：整段轨迹的 dx 标准差不低于 min_std。
    window>0：按时间窗口计算局部标准差，鼓励各段都有抖动（更易产生加速度起伏）。
    """
    dx = fake_seq[:, :, 0]  # (B, L)
    B, L = dx.shape
    if window is None or window <= 0 or L < window:
        std_per_sample = dx.std(dim=1).clamp(min=1e-8)
        return (min_std - std_per_sample).clamp(min=0).mean()
    patches = dx.unfold(dimension=1, size=window, step=1)  # (B, L-win+1, window)
    std_local = patches.std(dim=-1).clamp(min=1e-8)  # (B, L-win+1)
    return (min_std - std_local).clamp(min=0).mean()


def acceleration_strength_loss(fake_seq, min_acc_std=0.03):
    """
    加速度强度约束：鼓励「速度差分」(dx 的步间差) 有足够方差，
    使轨迹在时间上有加速/减速起伏，而不是近乎匀速（加速度接近 0）。
    fake_seq: (B, L, 2) 即 (dx, dy) 序列；速度 ≈ dx，加速度 ≈ dx 的一阶差分。
    """
    vel = fake_seq[:, :, 0]  # (B, L) dx
    acc = vel[:, 1:] - vel[:, :-1]  # (B, L-1) 步间加速度
    acc_flat = acc.reshape(acc.size(0), -1)
    acc_std = acc_flat.std(dim=1).clamp(min=1e-8)
    return (min_acc_std - acc_std).clamp(min=0).mean()


def drawdown_loss(fake_seq, margin_px: float = DRAWDOWN_MARGIN_PX):
    """
    回撤约束：惩罚累积位移 x(t)=cumsum(dx) 相对历史最大值的回撤。

    目的：你希望“整体上没有大幅度回撤”，允许极小的人类式回退（例如 1~2px），
    但不希望出现明显逆行/回撤段。

    fake_seq: (B, L, 2) 归一化空间
    margin_px: 允许的回撤阈值（像素）。超过该阈值的回撤将被惩罚（平方惩罚，强调大回撤）。
    """
    dx = fake_seq[:, :, 0]  # (B, L) 归一化 dx
    x = torch.cumsum(dx, dim=1)
    run_max = torch.cummax(x, dim=1).values
    dd = run_max - x
    margin = float(margin_px) / float(NORM_SCALE)
    dd_excess = (dd - margin).clamp(min=0.0)
    return torch.mean(dd_excess ** 2)


# ============================
# 训练主循环
# ============================
def train():
    parser = argparse.ArgumentParser(description="Train WGAN-GP (v2)")
    parser.add_argument("--data-dir", type=str, default="dataset", help="dataset 根目录")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32, help="较小 batch 以获得更多 G 更新")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=str, default="checkpoints", help="模型保存目录")
    parser.add_argument("--d-steps", type=int, default=D_STEPS)
    parser.add_argument("--lambda-gp", type=float, default=LAMBDA_GP)
    parser.add_argument("--lambda-geom", type=float, default=LAMBDA_GEOM)
    parser.add_argument("--lambda-smooth", type=float, default=LAMBDA_SMOOTH)
    parser.add_argument("--lambda-spread", type=float, default=LAMBDA_SPREAD)
    parser.add_argument("--lambda-jitter", type=float, default=LAMBDA_JITTER)
    parser.add_argument("--lambda-acc", type=float, default=LAMBDA_ACC, help="加速度强度约束权重")
    parser.add_argument("--min-acc-std", type=float, default=MIN_ACC_STD, help="速度差分的最小标准差")
    parser.add_argument("--jitter-window", type=int, default=JITTER_WINDOW, help="0=全局抖动; 8 等=局部窗口抖动")
    parser.add_argument("--lambda-drawdown", type=float, default=LAMBDA_DRAWDOWN)
    parser.add_argument("--drawdown-margin-px", type=float, default=DRAWDOWN_MARGIN_PX)
    args = parser.parse_args()

    # ---- 数据加载 ----
    train_dir = Path(args.data_dir) / "train"
    test_dir = Path(args.data_dir) / "test"
    train_samples = load_dataset(train_dir)
    test_samples = load_dataset(test_dir)
    if not train_samples:
        print("dataset/train 下无 JSON 轨迹文件，请先通过 web/index.html 采集并导出到 train/")
        return

    print(f"训练样本数: {len(train_samples)}, 测试样本数: {len(test_samples)}")
    print(f"序列长度: {SEQ_LEN}, 特征维度: 2 (dx, dy), 设备: {DEVICE}")
    print(f"超参: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, d_steps={args.d_steps}")
    print(
        f"  λ_gp={args.lambda_gp}, λ_geom={args.lambda_geom}, λ_smooth={args.lambda_smooth}, "
        f"λ_spread={args.lambda_spread}, λ_jitter={args.lambda_jitter}, λ_acc={args.lambda_acc} "
        f"(min_acc_std={args.min_acc_std}, jitter_win={args.jitter_window}), "
        f"λ_drawdown={args.lambda_drawdown} (margin={args.drawdown_margin_px}px)"
    )

    # ---- 模型 ----
    G = Generator(max_len=SEQ_LEN).to(DEVICE)
    D = Discriminator(max_len=SEQ_LEN).to(DEVICE)
    g_opt = Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    d_opt = Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.9))

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # ---- 训练 ----
    for epoch in range(args.epochs):
        G.train()
        D.train()
        d_loss_sum, g_loss_sum = 0.0, 0.0
        geom_sum, smooth_sum, spread_sum, jitter_sum, acc_sum, dd_sum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        n_d_steps, n_g_steps = 0, 0

        for step, (cond_np, seq_np) in enumerate(
            prepare_batches(train_samples, max_len=SEQ_LEN, batch_size=args.batch_size, shuffle=True)
        ):
            cond = torch.from_numpy(cond_np).float().to(DEVICE)
            real = torch.from_numpy(seq_np).float().to(DEVICE)
            B = real.size(0)
            z = torch.randn(B, LATENT_DIM, device=DEVICE)

            # ---- 判别器 ----
            fake = G(z, cond).detach()
            d_real = D(real, cond)
            d_fake = D(fake, cond)
            gp = gradient_penalty(D, real, fake, cond)
            d_loss = d_fake.mean() - d_real.mean() + args.lambda_gp * gp

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            d_loss_sum += d_loss.item()
            n_d_steps += 1

            # ---- 生成器（每 d_steps 步训练一次）----
            if step % args.d_steps == 0:
                z = torch.randn(B, LATENT_DIM, device=DEVICE)
                gen_seq = G(z, cond)

                loss_adv = -D(gen_seq, cond).mean()
                loss_geom = geometry_loss(gen_seq, cond)
                loss_smooth = smoothness_loss(gen_seq)
                loss_spread = spread_loss(gen_seq)
                loss_jitter = jitter_loss(
                    gen_seq, min_std=0.06, window=args.jitter_window if args.jitter_window > 0 else None
                )
                loss_acc = acceleration_strength_loss(gen_seq, min_acc_std=args.min_acc_std)
                loss_dd = drawdown_loss(gen_seq, margin_px=args.drawdown_margin_px)

                g_loss = (loss_adv
                          + args.lambda_geom * loss_geom
                          + args.lambda_smooth * loss_smooth
                          + args.lambda_spread * loss_spread
                          + args.lambda_jitter * loss_jitter
                          + args.lambda_acc * loss_acc
                          + args.lambda_drawdown * loss_dd)

                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()

                g_loss_sum += g_loss.item()
                geom_sum += loss_geom.item()
                smooth_sum += loss_smooth.item()
                spread_sum += loss_spread.item()
                jitter_sum += loss_jitter.item()
                acc_sum += loss_acc.item()
                dd_sum += loss_dd.item()
                n_g_steps += 1

        # ---- 日志 ----
        if (epoch + 1) % 100 == 0 or epoch == 0:
            d_avg = d_loss_sum / max(n_d_steps, 1)
            g_avg = g_loss_sum / max(n_g_steps, 1)
            geom_avg = geom_sum / max(n_g_steps, 1)
            smooth_avg = smooth_sum / max(n_g_steps, 1)
            spread_avg = spread_sum / max(n_g_steps, 1)
            jitter_avg = jitter_sum / max(n_g_steps, 1)
            dd_avg = dd_sum / max(n_g_steps, 1)
            acc_avg = acc_sum / max(n_g_steps, 1)
            print(
                f"Epoch {epoch+1}/{args.epochs}  "
                f"D={d_avg:.4f}  G={g_avg:.4f}  "
                f"Geom={geom_avg:.4f}  Smooth={smooth_avg:.6f}  "
                f"Spread={spread_avg:.4f}  Jitter={jitter_avg:.4f}  Acc={acc_avg:.4f}  Drawdown={dd_avg:.6f}"
            )

        # ---- 定期保存 ----
        if (epoch + 1) % 500 == 0:
            ckpt_path = Path(args.out) / f"wgan_epoch{epoch+1}.pt"
            torch.save({"G": G.state_dict(), "D": D.state_dict()}, ckpt_path)
            print(f"  checkpoint -> {ckpt_path}")

    # ---- 最终保存 ----
    final_path = Path(args.out) / "wgan.pt"
    torch.save({"G": G.state_dict(), "D": D.state_dict()}, final_path)
    print(f"训练完成，模型已保存: {final_path}")


if __name__ == "__main__":
    train()
