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

import matplotlib.pyplot as plt
import numpy as np

# ============================
# 自包含常量（与 dataset/preprocess.py 保持一致）
# ============================
SEQ_LEN = 128
NORM_SCALE = 10.0
D_MAX = 400.0


LATENT_DIM = 64


def _setup_matplotlib_chinese() -> None:
    """
    简单的 Matplotlib 中文显示设置，避免标题/坐标轴中文乱码。

    会尝试使用常见中文字体（不同操作系统名称略有差异）：
    - macOS: PingFang SC
    - Windows: Microsoft YaHei
    - Linux: SimHei
    """
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    # 解决负号显示为方框的问题
    plt.rcParams["axes.unicode_minus"] = False


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


def _compute_and_plot_kinematics(points: list[dict]) -> None:
    """基于离散 points 做简单的轨迹/速度/加速度/抖动统计与可视化。"""
    # 保证中文标题/标签不乱码
    _setup_matplotlib_chinese()

    if len(points) < 2:
        print("[WARN] 轨迹点过少，无法进行速度/加速度分析。")
        return

    t = np.array([p["t"] for p in points], dtype=np.float64)
    x = np.array([p["x"] for p in points], dtype=np.float64)
    y = np.array([p["y"] for p in points], dtype=np.float64)

    # 基本几何量
    dt = np.diff(t)
    dt_safe = np.where(dt <= 1e-6, 1e-6, dt)
    dx = np.diff(x)
    dy = np.diff(y)

    vx = dx / dt_safe
    vy = dy / dt_safe
    v = np.sqrt(vx**2 + vy**2)
    t_v = (t[:-1] + t[1:]) / 2.0

    # 加速度：对速度标量再求导
    if len(v) > 1:
        dv = np.diff(v)
        dt_a = np.diff(t_v)
        dt_a_safe = np.where(dt_a <= 1e-6, 1e-6, dt_a)
        a = dv / dt_a_safe
        t_a = (t_v[:-1] + t_v[1:]) / 2.0
    else:
        a = np.zeros(1, dtype=np.float64)
        t_a = np.array([t_v[0]], dtype=np.float64)

    # 轨迹长度 & 平均速度
    seg_dist = np.sqrt(dx**2 + dy**2)
    traj_len = float(np.sum(seg_dist))
    total_time = float(max(t[-1] - t[0], 1e-6))
    avg_speed = traj_len / total_time

    # 抖动：对 x 做简单平滑，观察残差的方差
    if len(x) >= 5:
        kernel_size = 5
        kernel = np.ones(kernel_size, dtype=np.float64) / kernel_size
        x_smooth = np.convolve(x, kernel, mode="same")
        jitter = x - x_smooth
    else:
        jitter = x * 0.0
    jitter_var = float(np.var(jitter))

    # === 绘图 ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    # 1) X - T（只画曲线，不标点）
    ax = axes[0]
    ax.plot(t, x, linewidth=1)
    ax.set_title("X - T")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("x (px)")

    # 2) 速度 - 时间（只画曲线，不标点）
    ax = axes[1]
    ax.plot(t_v, v, linewidth=1, color="tab:orange")
    ax.set_title("速度 - 时间")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("speed (px/ms)")

    # 3) 加速度 - 时间（只画曲线，不标点）
    ax = axes[2]
    ax.plot(t_a, a, linewidth=1, color="tab:green")
    ax.set_title("加速度 - 时间")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("acc (px/ms^2)")

    # 4) 平均速度 - 轨迹长度（单点散点图）
    ax = axes[3]
    ax.scatter([traj_len], [avg_speed], color="tab:blue")
    ax.set_title("平均速度 - 轨迹长度")
    ax.set_xlabel("轨迹长度 (px)")
    ax.set_ylabel("平均速度 (px/ms)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.annotate(
        f"L={traj_len:.1f}px\nv̄={avg_speed:.3f}px/ms",
        (traj_len, avg_speed),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=9,
    )

    # 5) 抖动/方差分析：x 抖动随时间
    ax = axes[4]
    ax.plot(t, jitter, linewidth=1, color="tab:red")
    ax.set_title(f"抖动分析 (var={jitter_var:.4f})")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("x 抖动 (px)")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    # 第 6 个子图可以留空，或简单展示文本摘要
    ax = axes[5]
    ax.axis("off")
    summary = (
        f"轨迹点数: {len(points)}\n"
        f"总时长: {total_time:.1f} ms\n"
        f"轨迹长度: {traj_len:.1f} px\n"
        f"平均速度: {avg_speed:.3f} px/ms\n"
        f"抖动方差: {jitter_var:.4f}"
    )
    ax.text(
        0.02,
        0.98,
        summary,
        va="top",
        ha="left",
        fontsize=10,
        transform=ax.transAxes,
    )

    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="ONNXRuntime trajectory demo")
    parser.add_argument("--onnx", type=str, default="checkpoints/GANSlider.onnx")
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

    # 生成轨迹后做可视化与统计分析
    _compute_and_plot_kinematics(points)


if __name__ == "__main__":
    main()

