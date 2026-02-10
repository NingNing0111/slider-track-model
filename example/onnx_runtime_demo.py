"""
ONNXRuntime 推理示例：根据目标距离生成滑块轨迹 points。

前置：
1) 先导出 ONNX 模型（默认输出到 resource/GANSlider.onnx）：
   uv run python export_onnx.py --checkpoint resource/GANSlider.pt --out resource/GANSlider.onnx

2) 安装 onnxruntime（若未安装）：
   uv add onnxruntime
   # 或 pip install onnxruntime

运行：
   uv run python example/onnx_runtime_demo.py --onnx resource/GANSlider.onnx --distance 180
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ============================
# 自包含常量（与 dataset/preprocess.py 保持一致）
# ============================
SEQ_LEN = 128
NORM_SCALE = 10.0
D_MAX = 400.0


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
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    *,
    seq_len: int = SEQ_LEN,
    norm_scale: float = NORM_SCALE,
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

    重要：
    - start_xy / end_xy 均为 px 单位的绝对坐标
    - 输出 points 的 x/y 也为 px 单位的绝对坐标
    """
    if seed is not None:
        np.random.seed(seed)

    seq_norm = np.asarray(seq_norm, dtype=np.float64)
    assert seq_norm.shape == (int(seq_len), 2), f"期望形状 ({int(seq_len)},2)，实际 {seq_norm.shape}"

    dx = seq_norm[:, 0] * float(norm_scale)
    dy = seq_norm[:, 1] * float(norm_scale)

    if add_jitter and jitter_std_px > 0:
        dx = dx + _colored_noise(len(dx), std=jitter_std_px, alpha=jitter_alpha)
        dy = dy + _colored_noise(len(dy), std=jitter_std_px * 0.5, alpha=jitter_alpha)

    if clip_dx_min_px is not None:
        dx = np.maximum(dx, float(clip_dx_min_px))
    if clip_dx_max_px is not None:
        dx = np.minimum(dx, float(clip_dx_max_px))

    # ---- 先在“相对坐标系”里累加（起点视为 0,0）----
    x_rel = np.cumsum(np.concatenate([[0.0], dx]))
    y_rel = np.cumsum(np.concatenate([[0.0], dy]))
    total_x = float(x_rel[-1])

    x0, y0 = float(start_xy[0]), float(start_xy[1])
    x1, y1 = float(end_xy[0]), float(end_xy[1])
    target_dx = x1 - x0
    target_dy = y1 - y0

    # ---- X 方向：缩放到 target_dx，并保证终点严格对齐 ----
    if abs(target_dx) < 1e-9:
        # 目标 x 不动：直接把 x 设置为常数
        x_rel = np.zeros_like(x_rel)
    else:
        # 生成器训练的是“向前”位移；推理时按方向进行镜像
        sign = 1.0 if target_dx >= 0 else -1.0
        # 先把生成的 x 相对位移变为正向（只影响方向，不改形状）
        x_rel_signed = x_rel * sign
        total_x_signed = float(x_rel_signed[-1])
        if total_x_signed < 1e-6:
            # 极端情况：输出几乎全 0，退化为线性插值
            x_rel_signed = np.linspace(0.0, abs(target_dx), len(x_rel_signed))
        else:
            x_rel_signed = x_rel_signed * (abs(target_dx) / total_x_signed)
        x_rel = x_rel_signed * sign

    # ---- Y 方向：保持模型形状 + 线性纠偏，使终点对齐到 target_dy ----
    # 这样既能保持抖动/曲线形态，又能保证最后一点 y == y1
    y_end = float(y_rel[-1])
    if len(y_rel) > 1:
        corr = np.linspace(0.0, target_dy - y_end, len(y_rel))
        y_rel = y_rel + corr
    else:
        y_rel = np.array([0.0], dtype=np.float64)

    if force_monotone_x:
        x_rel = np.maximum.accumulate(x_rel)
        end = float(x_rel[-1])
        if abs(target_dx) > 1e-9 and abs(end) > 1e-9:
            x_rel = x_rel * (target_dx / end)

    # ---- 转回绝对坐标（px）----
    x = x0 + x_rel
    y = y0 + y_rel
    # 再次强制终点精确对齐（避免浮点误差）
    x[-1] = x1
    y[-1] = y1

    n = len(x)
    total_time = _estimate_duration_ms(float(abs(target_dx)))
    t = np.linspace(0.0, total_time, n)
    jitter_t = np.zeros(n)
    if n > 2:
        jitter_t[1:-1] = np.random.uniform(-1.0, 1.0, size=n - 2)
    t = t + jitter_t
    t = np.maximum.accumulate(t)
    t[0] = 0.0

    return [{"x": float(x[i]), "y": float(y[i]), "t": float(t[i])} for i in range(n)]


def run_onnx(
    onnx_path: str,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    *,
    d_max: float = D_MAX,
    seq_len: int = SEQ_LEN,
    norm_scale: float = NORM_SCALE,
    seed: int | None = 42,
) -> list[dict]:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    if seed is not None:
        np.random.seed(seed)

    x0, y0 = float(start_xy[0]), float(start_xy[1])
    x1, y1 = float(end_xy[0]), float(end_xy[1])
    target_dx = x1 - x0
    # 条件是“目标距离”（训练时为正值范围），这里用 abs(dx) 做归一化
    d_norm = float(np.clip(abs(target_dx) / float(d_max), 0.0, 1.0))
    z = np.random.randn(1, LATENT_DIM).astype(np.float32)
    cond = np.array([[d_norm]], dtype=np.float32)

    seq = sess.run(["seq"], {"z": z, "cond": cond})[0]  # (1, L, 2)
    seq = np.asarray(seq, dtype=np.float32)[0]
    return seq_to_points(
        seq,
        start_xy=(x0, y0),
        end_xy=(x1, y1),
        seq_len=seq_len,
        norm_scale=norm_scale,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNXRuntime trajectory demo")
    parser.add_argument("--onnx", type=str, default="resource/GANSlider.onnx")
    parser.add_argument("--x0", type=float, default=0.0, help="起点 x（px）")
    parser.add_argument("--y0", type=float, default=0.0, help="起点 y（px）")
    parser.add_argument("--x1", type=float, default=180.0, help="终点 x（px）")
    parser.add_argument("--y1", type=float, default=0.0, help="终点 y（px）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="example/out_trajectory.json")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN, help="模型输出序列长度（默认 128）")
    parser.add_argument("--d-max", type=float, default=D_MAX, help="距离归一化用的 D_MAX（默认 400）")
    parser.add_argument("--norm-scale", type=float, default=NORM_SCALE, help="dx/dy 反归一化尺度（默认 10）")
    args = parser.parse_args()

    points = run_onnx(
        args.onnx,
        start_xy=(float(args.x0), float(args.y0)),
        end_xy=(float(args.x1), float(args.y1)),
        d_max=float(args.d_max),
        seq_len=int(args.seq_len),
        norm_scale=float(args.norm_scale),
        seed=args.seed,
    )
    obj = {
        "start": {"x": float(args.x0), "y": float(args.y0)},
        "end": {"x": float(args.x1), "y": float(args.y1)},
        "targetDistance": float(abs(float(args.x1) - float(args.x0))),
        "points": points,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 已生成轨迹: {out_path} (points={len(points)})")


if __name__ == "__main__":
    main()

