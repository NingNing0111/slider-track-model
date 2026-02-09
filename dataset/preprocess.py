"""预处理：重采样 + 归一化（参考 v2 训练思路，去掉 dt，统一缩放）。"""
import numpy as np

# ---- 全局常量 ----
SEQ_LEN = 128           # 重采样后固定长度（比原 100 更高分辨率）
DATA_DIM = 2            # 输出维度: (dx, dy)，不再含 dt
NORM_SCALE = 10.0       # dx, dy 统一缩放因子
D_MAX = 400.0           # 最大目标距离（归一化用）
DT_MS = 10.0            # 推理时每步假定间隔 10ms

# 兼容旧代码引用（inference.py 等）
NORM_DX_SCALE = NORM_SCALE
NORM_DY_SCALE = NORM_SCALE
NORM_DT_SCALE = DT_MS


def resample_trajectory(points, target_len=SEQ_LEN):
    """
    将原始轨迹重采样为 target_len 个等时间间隔增量。
    核心思路：按时间均匀插值 → 计算增量 → 归一化。

    参数:
        points: [{x, y, t}, ...]  页面导出的原始点
        target_len: 重采样后的序列长度

    返回:
        (seq, cond)
        - seq: ndarray (target_len, 2) 归一化 (dx, dy) 增量
        - cond: ndarray (1,) 归一化目标距离
        如果轨迹无效则返回 None
    """
    if len(points) < 2:
        return None

    t = np.array([p["t"] for p in points], dtype=np.float64)
    x = np.array([p["x"] for p in points], dtype=np.float64)
    y = np.array([p["y"] for p in points], dtype=np.float64)

    total_time = t[-1] - t[0]
    if total_time <= 0:
        return None

    # 均匀时间轴，target_len+1 个点 → target_len 个增量
    new_t = np.linspace(t[0], t[-1], target_len + 1)
    new_x = np.interp(new_t, t, x)
    new_y = np.interp(new_t, t, y)

    dx = new_x[1:] - new_x[:-1]
    dy = new_y[1:] - new_y[:-1]

    seq = np.stack([dx, dy], axis=1) / NORM_SCALE   # (target_len, 2)

    target_dist = np.clip((x[-1] - x[0]) / D_MAX, 0.0, 1.0)

    return np.array([target_dist], dtype=np.float32), seq.astype(np.float32)


def sample_to_sequence(sample, max_len=SEQ_LEN):
    """
    (D, points) -> (condition, seq)  使用重采样。
    condition: [D_norm]
    seq: (max_len, 2) 归一化后的 (dx, dy) 增量
    """
    D, points = sample
    return resample_trajectory(points, target_len=max_len)


def prepare_batches(samples, max_len=SEQ_LEN, batch_size=32, shuffle=True):
    """samples = [(D, points), ...] -> 迭代 (cond_batch, seq_batch)。"""
    parsed = []
    for s in samples:
        r = sample_to_sequence(s, max_len=max_len)
        if r is not None:
            parsed.append(r)
    if not parsed:
        return
    conds = np.stack([p[0] for p in parsed])
    seqs = np.stack([p[1] for p in parsed])
    n = len(conds)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = idx[start:end]
        yield conds[batch_idx], seqs[batch_idx]
