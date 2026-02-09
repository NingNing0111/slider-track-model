"""
推理：根据目标距离生成一条轨迹（v2: 2D 模型输出 dx/dy，时间用固定间隔重建）。
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


def _target_duration_ms(target_distance):
    """人工轨迹典型时长（ms）：距离越长略增，约 80~200ms。"""
    return 80 + 0.35 * min(target_distance, 400)


def _colored_noise(n: int, std: float, alpha: float = 0.85) -> np.ndarray:
    """
    生成一段一阶自回归(AR(1))的“有相关性噪声”，更像手部微抖而非白噪声。
    - alpha 越大相关性越强、越“平滑”
    - std 为输出标准差（像素）
    """
    if n <= 0 or std <= 0:
        return np.zeros(max(n, 0), dtype=np.float64)
    eps = np.random.normal(0, 1.0, size=n).astype(np.float64)
    x = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        x[i] = alpha * x[i - 1] + eps[i]
    s = np.std(x)
    if s > 1e-9:
        x = x / s * std
    else:
        x = x * 0.0
    return x


def generate_trajectory(
    G,
    target_distance,
    seed=None,
    scale_time_to_human=True,
    warp_time_front_heavy=True,
    add_jitter=True,
    jitter_std_px=0.9,
    jitter_alpha=0.85,
    ensure_monotonic_time=True,
):
    """
    给定目标距离（像素），生成一条轨迹 points: [{x, y, t}, ...]。
    v2: 模型只输出 (dx, dy)，时间由固定间隔 DT_MS 构建，再缩放到人工时长范围。
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

    # 可选：加小幅抖动
    if add_jitter and jitter_std_px > 0:
        # 用相关噪声模拟“手抖”，避免白噪声那种不真实的高频毛刺
        dx = dx + _colored_noise(len(dx), std=jitter_std_px, alpha=jitter_alpha)
        dy = dy + _colored_noise(len(dy), std=jitter_std_px * 0.5, alpha=jitter_alpha)

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

    # 时间：每步固定 DT_MS（10ms），加随机抖动 ±1ms
    n_step = len(dx)
    dt_arr = np.full(n_step, DT_MS) + np.random.uniform(-1.0, 1.0, size=n_step)
    dt_arr = np.maximum(dt_arr, 1.0)
    t = np.cumsum(np.concatenate([[0], dt_arr]))

    # 缩放总时长到人工典型范围
    if scale_time_to_human and t[-1] > 1e-6:
        target_t = _target_duration_ms(target_distance)
        if warp_time_front_heavy:
            # 按位移比例重映射时间，前段快、后段慢
            x_frac = (x - x[0]) / (x[-1] - x[0] + 1e-9)
            t_frac = np.power(x_frac, 2.0)
            t = t_frac * target_t
        else:
            t = t / t[-1] * target_t

        # 时间必须单调递增，否则 dt≈0 会导致速度/加速度计算出现极大尖峰
        if ensure_monotonic_time:
            t = np.asarray(t, dtype=np.float64)
            # 先去掉可能的回退/重复
            t = np.maximum.accumulate(t)
            # 再强制严格递增（微小 epsilon 递增），最后重新缩放到 target_t
            eps = 1e-3  # ms
            t = np.maximum.accumulate(t + np.arange(len(t), dtype=np.float64) * eps)
            denom = (t[-1] - t[0])
            if denom > 1e-9:
                t = (t - t[0]) / denom * target_t

    n = len(x)
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
