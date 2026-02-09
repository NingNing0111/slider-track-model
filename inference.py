"""
推理：根据目标距离生成一条轨迹（v2: 2D 模型输出 dx/dy，时间用等间隔重建）。

核心思想：
  训练时通过等时间间隔重采样把速度信息编码进 dx 分布，
  推理时只需还原为等间隔时间轴即可，不需要额外 warp。
"""
from pathlib import Path
import json
import torch
import numpy as np
from dataset.preprocess import NORM_SCALE, D_MAX, DT_MS, SEQ_LEN
from model.wgan import Generator

LATENT_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_generator(checkpoint_path="checkpoints/wgan.pt"):
    G = Generator(max_len=SEQ_LEN).to(DEVICE)
    ckpt = torch.load(Path(checkpoint_path), map_location=DEVICE, weights_only=True)
    if "G" in ckpt:
        G.load_state_dict(ckpt["G"])
    else:
        G.load_state_dict(ckpt)
    G.eval()
    return G


def _estimate_duration_ms(target_distance):
    """
    估算人工轨迹典型总时长（ms）。
    从采集数据观察：200px ≈ 1200~1800ms，50px ≈ 600~1000ms。
    """
    return max(400, 600 + 4.5 * min(target_distance, 400))


def _colored_noise(n: int, std: float, alpha: float = 0.85) -> np.ndarray:
    """AR(1) 相关噪声，模拟手部微抖。"""
    if n <= 0 or std <= 0:
        return np.zeros(max(n, 0), dtype=np.float64)
    eps = np.random.normal(0, 1.0, size=n).astype(np.float64)
    x = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        x[i] = alpha * x[i - 1] + eps[i]
    s = np.std(x)
    if s > 1e-9:
        x = x / s * std
    return x


def generate_trajectory(
    G,
    target_distance,
    seed=None,
    add_jitter=True,
    jitter_std_px=0.8,
    jitter_alpha=0.85,
    clip_dx_min_px=-2.0,
    clip_dx_max_px=None,
    force_monotone_x=False,
):
    """
    给定目标距离（像素），生成一条轨迹 points: [{x, y, t}, ...]。

    时间重建策略：
      训练数据已通过等时间间隔重采样，dx 本身编码了速度信息。
      推理时用等间隔时间轴 + 缩放到人工典型总时长即可。
      不做 power warp（会破坏 dx 已编码的速度分布）。
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    D_norm = min(1.0, max(0.0, target_distance / D_MAX))
    cond = torch.tensor([[D_norm]], dtype=torch.float32, device=DEVICE)
    z = torch.randn(1, LATENT_DIM, device=DEVICE)

    with torch.no_grad():
        seq = G(z, cond).cpu().numpy()[0]   # (SEQ_LEN, 2)

    # 反归一化
    dx = seq[:, 0] * NORM_SCALE
    dy = seq[:, 1] * NORM_SCALE

    # 可选：加相关噪声模拟手抖
    if add_jitter and jitter_std_px > 0:
        dx = dx + _colored_noise(len(dx), std=jitter_std_px, alpha=jitter_alpha)
        dy = dy + _colored_noise(len(dy), std=jitter_std_px * 0.5, alpha=jitter_alpha)

    # ---- 约束：限制每步回退与尖峰（在终点缩放前做，避免缩放放大回撤）----
    if clip_dx_min_px is not None:
        dx = np.maximum(dx, float(clip_dx_min_px))
    if clip_dx_max_px is not None:
        dx = np.minimum(dx, float(clip_dx_max_px))

    # 累加位移
    x = np.cumsum(np.concatenate([[0], dx]))
    total_x = x[-1]

    # 终点缩放到目标距离
    if total_x < 1e-6:
        n_step = max(1, len(dx))
        dx = np.ones(n_step, dtype=np.float64) * (target_distance / n_step)
    else:
        dx = dx * (target_distance / total_x)

    x = np.cumsum(np.concatenate([[0], dx]))
    y = np.cumsum(np.concatenate([[0], dy]))

    # 可选：强制单调（彻底消除回撤），再做一次终点对齐
    if force_monotone_x:
        x = np.maximum.accumulate(x)
        end = float(x[-1])
        if end > 1e-9:
            x = x * (float(target_distance) / end)
        dx = np.diff(x)

    # ---- 时间重建：等间隔 + 小抖动，缩放到人工典型总时长 ----
    n = len(x)
    total_time = _estimate_duration_ms(target_distance)
    # 等间隔时间轴 + ±1ms 随机抖动
    t = np.linspace(0, total_time, n)
    jitter_t = np.zeros(n)
    jitter_t[1:-1] = np.random.uniform(-1.0, 1.0, size=n - 2)
    t = t + jitter_t
    # 保证单调递增
    t = np.maximum.accumulate(t)
    t[0] = 0.0

    points = [{"x": float(x[i]), "y": float(y[i]), "t": float(t[i])} for i in range(n)]
    return points


def generate_and_save(checkpoint_path, target_distance, out_path, seed=None):
    G = load_generator(checkpoint_path)
    points = generate_trajectory(G, target_distance, seed=seed)
    obj = {"targetDistance": target_distance, "points": points}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return obj
