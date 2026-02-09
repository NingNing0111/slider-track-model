"""
ONNXRuntime 推理示例：根据目标距离生成滑块轨迹 points。

前置：
1) 先导出 ONNX 模型（默认输出到 checkpoints/wgan.onnx）：
   uv run python export_onnx.py --checkpoint checkpoints/wgan.pt --out checkpoints/wgan.onnx

2) 安装 onnxruntime（若未安装）：
   uv add onnxruntime
   # 或 pip install onnxruntime

运行：
   uv run python example/onnx_runtime_demo.py --onnx checkpoints/wgan.onnx --distance 180
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dataset.preprocess import NORM_SCALE, D_MAX, SEQ_LEN


LATENT_DIM = 64


def _estimate_duration_ms(target_distance: float) -> float:
    # 与 inference.py 保持一致的经验估计
    return float(max(400, 600 + 4.5 * min(target_distance, 400)))


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


def seq_to_points(
    seq_norm: np.ndarray,
    target_distance: float,
    seed: int | None = None,
    add_jitter: bool = True,
    jitter_std_px: float = 0.8,
    jitter_alpha: float = 0.85,
    clip_dx_min_px: float = -2.0,
    clip_dx_max_px: float | None = None,
    force_monotone_x: bool = False,
) -> list[dict]:
    """
    将 ONNX 输出的归一化序列 (SEQ_LEN, 2) 还原为 points: [{x,y,t}, ...]。
    逻辑与 inference.generate_trajectory 的后处理保持一致。
    """
    if seed is not None:
        np.random.seed(seed)

    seq_norm = np.asarray(seq_norm, dtype=np.float64)
    assert seq_norm.shape == (SEQ_LEN, 2), f"期望形状 ({SEQ_LEN},2)，实际 {seq_norm.shape}"

    dx = seq_norm[:, 0] * float(NORM_SCALE)
    dy = seq_norm[:, 1] * float(NORM_SCALE)

    if add_jitter and jitter_std_px > 0:
        dx = dx + _colored_noise(len(dx), std=jitter_std_px, alpha=jitter_alpha)
        dy = dy + _colored_noise(len(dy), std=jitter_std_px * 0.5, alpha=jitter_alpha)

    if clip_dx_min_px is not None:
        dx = np.maximum(dx, float(clip_dx_min_px))
    if clip_dx_max_px is not None:
        dx = np.minimum(dx, float(clip_dx_max_px))

    x = np.cumsum(np.concatenate([[0.0], dx]))
    total_x = float(x[-1])

    if total_x < 1e-6:
        n_step = max(1, len(dx))
        dx = np.ones(n_step, dtype=np.float64) * (float(target_distance) / n_step)
    else:
        dx = dx * (float(target_distance) / total_x)

    x = np.cumsum(np.concatenate([[0.0], dx]))
    y = np.cumsum(np.concatenate([[0.0], dy]))

    if force_monotone_x:
        x = np.maximum.accumulate(x)
        end = float(x[-1])
        if end > 1e-9:
            x = x * (float(target_distance) / end)

    n = len(x)
    total_time = _estimate_duration_ms(float(target_distance))
    t = np.linspace(0.0, total_time, n)
    jitter_t = np.zeros(n)
    if n > 2:
        jitter_t[1:-1] = np.random.uniform(-1.0, 1.0, size=n - 2)
    t = t + jitter_t
    t = np.maximum.accumulate(t)
    t[0] = 0.0

    return [{"x": float(x[i]), "y": float(y[i]), "t": float(t[i])} for i in range(n)]


def run_onnx(onnx_path: str, target_distance: float, seed: int | None = 42) -> list[dict]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    if seed is not None:
        np.random.seed(seed)

    d_norm = float(np.clip(target_distance / float(D_MAX), 0.0, 1.0))
    z = np.random.randn(1, LATENT_DIM).astype(np.float32)
    cond = np.array([[d_norm]], dtype=np.float32)

    seq = sess.run(["seq"], {"z": z, "cond": cond})[0]  # (1, L, 2)
    seq = np.asarray(seq, dtype=np.float32)[0]
    return seq_to_points(seq, target_distance=target_distance, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNXRuntime trajectory demo")
    parser.add_argument("--onnx", type=str, default="checkpoints/wgan.onnx")
    parser.add_argument("--distance", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="example/out_trajectory.json")
    args = parser.parse_args()

    points = run_onnx(args.onnx, target_distance=args.distance, seed=args.seed)
    obj = {"targetDistance": float(args.distance), "points": points}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 已生成轨迹: {out_path} (points={len(points)})")


if __name__ == "__main__":
    main()

