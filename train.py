"""
训练脚本（v2 思路）：
- 重采样预处理（等时间间隔插值，固定序列长度 128）
- 2D 输出 (dx, dy)，去掉 dt
- WGAN-GP：D 训练 5 次，G 训练 1 次
- 辅助 Loss：geometry_loss（终点约束）+ smoothness_loss（平滑约束）
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
LAMBDA_GEOM = 5.0       # 几何约束（终点距离）
LAMBDA_SMOOTH = 2.0     # 平滑约束（抗尖刺）
D_STEPS = 5             # 判别器每训练 D_STEPS 次，生成器训练 1 次
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
    total_dx = fake_seq[:, :, 0].sum(dim=1)                   # 归一化空间的总 dx
    target_val = cond.squeeze(1) * (D_MAX / NORM_SCALE)        # 转到同空间
    return F.mse_loss(total_dx, target_val)


def smoothness_loss(fake_seq):
    """
    平滑约束：惩罚加加速度 (Jerk)，消除生成轨迹的尖刺。
    fake_seq: (B, L, 2)
    """
    vel_diff = fake_seq[:, 1:, :] - fake_seq[:, :-1, :]       # 一阶差分 (加速度)
    acc_diff = vel_diff[:, 1:, :] - vel_diff[:, :-1, :]        # 二阶差分 (Jerk)
    return torch.mean(acc_diff ** 2)


# ============================
# 训练主循环
# ============================
def train():
    parser = argparse.ArgumentParser(description="Train WGAN-GP (v2)")
    parser.add_argument("--data-dir", type=str, default="dataset", help="dataset 根目录")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", type=str, default="checkpoints", help="模型保存目录")
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
    print(f"超参: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    print(f"  LAMBDA_GP={LAMBDA_GP}, LAMBDA_GEOM={LAMBDA_GEOM}, LAMBDA_SMOOTH={LAMBDA_SMOOTH}, D_STEPS={D_STEPS}")

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
        d_loss_sum, g_loss_sum, geom_sum, smooth_sum = 0.0, 0.0, 0.0, 0.0
        n_d_steps, n_g_steps = 0, 0

        for step, (cond_np, seq_np) in enumerate(
            prepare_batches(train_samples, max_len=SEQ_LEN, batch_size=args.batch_size, shuffle=True)
        ):
            cond = torch.from_numpy(cond_np).float().to(DEVICE)
            real = torch.from_numpy(seq_np).float().to(DEVICE)
            B = real.size(0)
            z = torch.randn(B, LATENT_DIM, device=DEVICE)

            # ---- 判别器 (每步都训练) ----
            fake = G(z, cond).detach()
            d_real = D(real, cond)
            d_fake = D(fake, cond)
            gp = gradient_penalty(D, real, fake, cond)
            d_loss = d_fake.mean() - d_real.mean() + LAMBDA_GP * gp

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            d_loss_sum += d_loss.item()
            n_d_steps += 1

            # ---- 生成器 (每 D_STEPS 步训练一次) ----
            if step % D_STEPS == 0:
                z = torch.randn(B, LATENT_DIM, device=DEVICE)
                gen_seq = G(z, cond)

                loss_adv = -D(gen_seq, cond).mean()
                loss_geom = geometry_loss(gen_seq, cond)
                loss_smooth = smoothness_loss(gen_seq)

                g_loss = loss_adv + LAMBDA_GEOM * loss_geom + LAMBDA_SMOOTH * loss_smooth

                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()

                g_loss_sum += g_loss.item()
                geom_sum += loss_geom.item()
                smooth_sum += loss_smooth.item()
                n_g_steps += 1

        # ---- 日志 ----
        if (epoch + 1) % 100 == 0 or epoch == 0:
            d_avg = d_loss_sum / max(n_d_steps, 1)
            g_avg = g_loss_sum / max(n_g_steps, 1)
            geom_avg = geom_sum / max(n_g_steps, 1)
            smooth_avg = smooth_sum / max(n_g_steps, 1)
            print(
                f"Epoch {epoch+1}/{args.epochs}  "
                f"D={d_avg:.4f}  G={g_avg:.4f}  "
                f"Geom={geom_avg:.4f}  Smooth={smooth_avg:.6f}"
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
