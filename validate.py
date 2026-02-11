"""
滑块轨迹人工行为校验模块。

对单条轨迹计算核心观察指标，并可基于人工数据统计的阈值做通过/不通过判定，
用于训练后自检或推理时过滤不合格轨迹。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def points_to_arrays(points: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """points [{x,y,t}, ...] -> t, x, y 数组。"""
    t = np.array([p["t"] for p in points], dtype=float)
    x = np.array([p["x"] for p in points], dtype=float)
    y = np.array([p["y"] for p in points], dtype=float)
    return t, x, y


def compute_trajectory_metrics(
    points: list[dict],
    *,
    jitter_window: int = 5,
) -> dict[str, float]:
    """
    计算单条轨迹的核心观察指标，供人工行为校验使用。

    返回字段说明：
    - total_time_ms: 总时长 (ms)
    - total_distance_x: X 方向总位移 (px)，应接近 targetDistance
    - total_distance_path: 路径总长 (px)
    - mean_velocity: 平均速度 (路径长/总时间, px/ms)
    - max_velocity: 最大瞬时速度 (px/ms)
    - velocity_std: 速度标准差（过大可能过于抖动，过小可能过于平滑）
    - acceleration_mean, acceleration_std: 加速度均值与标准差
    - max_acceleration: 最大加速度绝对值（过大像瞬移）
    - jitter_mean: 滑动窗口位移方差的均值（模拟手抖程度）
    - y_range: Y 方向位移范围 (px)，水平滑块应较小但不为 0
    - max_drawdown_px: X 方向最大回撤 (px)，即相对前高的最大回退
    - num_points: 采样点数
    """
    if len(points) < 2:
        return {
            "total_time_ms": 0.0,
            "total_distance_x": 0.0,
            "total_distance_path": 0.0,
            "mean_velocity": 0.0,
            "max_velocity": 0.0,
            "velocity_std": 0.0,
            "acceleration_mean": 0.0,
            "acceleration_std": 0.0,
            "max_acceleration": 0.0,
            "jitter_mean": 0.0,
            "y_range": 0.0,
            "max_drawdown_px": 0.0,
            "num_points": len(points),
        }

    t, x, y = points_to_arrays(points)
    dt = np.diff(t)
    dt = np.maximum(dt, 1e-6)
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    v = ds / dt
    t_mid = (t[:-1] + t[1:]) / 2

    total_time_ms = float(t[-1] - t[0])
    total_distance_x = float(x[-1] - x[0])
    total_distance_path = float(np.sum(ds))
    mean_velocity = total_distance_path / total_time_ms if total_time_ms > 0 else 0.0
    max_velocity = float(np.max(v)) if len(v) > 0 else 0.0
    velocity_std = float(np.std(v)) if len(v) > 1 else 0.0

    if len(v) >= 2:
        dv = np.diff(v)
        dt_mid = np.diff(t_mid)
        dt_mid = np.maximum(dt_mid, 1e-6)
        a = dv / dt_mid
        acceleration_mean = float(np.mean(np.abs(a)))
        acceleration_std = float(np.std(a))
        max_acceleration = float(np.max(np.abs(a)))
    else:
        acceleration_mean = acceleration_std = max_acceleration = 0.0

    # 抖动：滑动窗口内位移的方差
    n = len(ds)
    jitter = np.zeros(n)
    half = jitter_window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        jitter[i] = np.var(ds[lo:hi]) if hi > lo else 0
    jitter_mean = float(np.mean(jitter))

    y_range = float(np.max(y) - np.min(y))
    run_max = np.maximum.accumulate(x)
    drawdown = run_max - x
    max_drawdown_px = float(np.max(drawdown))

    return {
        "total_time_ms": total_time_ms,
        "total_distance_x": total_distance_x,
        "total_distance_path": total_distance_path,
        "mean_velocity": mean_velocity,
        "max_velocity": max_velocity,
        "velocity_std": velocity_std,
        "acceleration_mean": acceleration_mean,
        "acceleration_std": acceleration_std,
        "max_acceleration": max_acceleration,
        "jitter_mean": jitter_mean,
        "y_range": y_range,
        "max_drawdown_px": max_drawdown_px,
        "num_points": len(points),
    }


def build_human_stats(
    trajectories: list[list[dict]],
    *,
    jitter_window: int = 5,
    low_percentile: float = 5.0,
    high_percentile: float = 95.0,
) -> dict[str, Any]:
    """
    从多个人工轨迹计算各指标的百分位，作为校验阈值。

    trajectories: 列表，每个元素为 points: [{x,y,t}, ...]
    返回: 每个指标有 "p_low", "p_high", "mean", "std"，用于后续判定。
    """
    if not trajectories:
        return {}

    keys = [
        "total_time_ms",
        "total_distance_x",
        "mean_velocity",
        "max_velocity",
        "velocity_std",
        "acceleration_mean",
        "acceleration_std",
        "max_acceleration",
        "jitter_mean",
        "y_range",
        "max_drawdown_px",
    ]
    by_key: dict[str, list[float]] = {k: [] for k in keys}

    for pts in trajectories:
        m = compute_trajectory_metrics(pts, jitter_window=jitter_window)
        for k in keys:
            if k in m:
                by_key[k].append(m[k])

    stats = {}
    for k, values in by_key.items():
        if not values:
            continue
        arr = np.array(values)
        stats[k] = {
            "p_low": float(np.percentile(arr, low_percentile)),
            "p_high": float(np.percentile(arr, high_percentile)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "count": len(values),
        }
    return stats


def check_trajectory_pass(
    points: list[dict],
    human_stats: dict[str, Any],
    *,
    tolerance: float = 1.5,
    require_keys: list[str] | None = None,
) -> tuple[bool, list[str], dict[str, float]]:
    """
    判断单条轨迹是否落在人工统计的合理范围内。

    - human_stats: 由 build_human_stats 得到，或从 JSON 加载。
    - tolerance: 允许超出 p_low/p_high 的倍数（例如 1.5 表示最多放宽到 1.5 倍范围）。
    - require_keys: 必须参与判定的指标；默认使用全部存在阈值的指标。

    返回: (是否通过, 不通过原因列表, 当前轨迹指标)
    """
    metrics = compute_trajectory_metrics(points)
    failures: list[str] = []

    if require_keys is None:
        require_keys = [k for k in metrics if k in human_stats and "p_low" in human_stats[k]]

    for k in require_keys:
        if k not in human_stats or "p_low" not in human_stats[k]:
            continue
        s = human_stats[k]
        v = metrics.get(k, 0.0)
        low = s["p_low"]
        high = s["p_high"]
        span = high - low
        if span <= 0:
            span = max(abs(low), abs(high), 1e-6)
        margin = (tolerance - 1.0) * span * 0.5
        low_ext = low - margin
        high_ext = high + margin
        if v < low_ext:
            failures.append(f"{k}={v:.3f} 低于人工范围 [{low:.3f}, {high:.3f}]")
        elif v > high_ext:
            failures.append(f"{k}={v:.3f} 高于人工范围 [{low:.3f}, {high:.3f}]")

    return (len(failures) == 0, failures, metrics)


def load_human_stats_from_dataset_dir(data_dir: str | Path) -> dict[str, Any]:
    """从 dataset/train 或指定目录下所有 JSON 加载轨迹并构建 human_stats。"""
    from dataset.io import load_dataset

    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return {}
    samples = load_dataset(data_dir)
    trajectories = [pts for _, pts in samples]
    return build_human_stats(trajectories)


def save_human_stats(stats: dict[str, Any], path: str | Path) -> None:
    """将 human_stats 保存为 JSON，便于复现与调参。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def load_human_stats(path: str | Path) -> dict[str, Any]:
    """从 JSON 加载 human_stats。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    """示例：从 dataset/train 构建 human_stats，并校验一条轨迹（若存在 compare 或 test）。"""
    import sys
    from dataset.io import load_dataset

    train_dir = Path("dataset/train")
    if not train_dir.is_dir():
        print("未找到 dataset/train，请先准备训练集。")
        sys.exit(1)

    samples = load_dataset(train_dir)
    trajectories = [pts for _, pts in samples]
    print(f"从 {train_dir} 加载 {len(trajectories)} 条轨迹")
    if not trajectories:
        sys.exit(1)

    stats = build_human_stats(trajectories)
    out = Path("dataset/compare/human_stats.json")
    save_human_stats(stats, out)
    print(f"已保存人工统计到 {out}")

    # 若存在 compare 的模型轨迹，做一次校验示例
    model_json = Path("dataset/compare/model.json")
    if model_json.exists():
        with open(model_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        pts = data.get("points") or (data.get("trajectories", [{}])[0].get("points"))
        if pts:
            passed, failures, metrics = check_trajectory_pass(pts, stats)
            print("模型轨迹校验:", "通过" if passed else "不通过")
            for f in failures:
                print("  -", f)
            print("指标摘要:", {k: round(v, 4) for k, v in list(metrics.items())[:6]})


if __name__ == "__main__":
    main()
