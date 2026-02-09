"""
对比绘制：dataset/compare/index.json（页面导出）与模型生成轨迹的对比图。
包含：轨迹 X-T、速度-时间、加速度-时间、平均速度-轨迹长度、抖动/方差分析。
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互后端，兼容无 GUI 服务器
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager

from inference import load_generator, generate_trajectory

# 自动检测可用中文字体，找不到则使用英文标签
_CJK_CANDIDATES = [
    "SimHei", "Microsoft YaHei",                       # Windows
    "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",       # Linux 常见
    "Noto Sans CJK SC", "Noto Sans SC",                # Google Noto
    "Source Han Sans SC",                               # Adobe 思源
    "AR PL UMing CN", "AR PL UKai CN",                 # 文鼎
]
_USE_CN = False
for _fname in _CJK_CANDIDATES:
    if font_manager.findfont(_fname, fallback_to_default=False):
        rcParams["font.sans-serif"] = [_fname, "DejaVu Sans"]
        _USE_CN = True
        break
if not _USE_CN:
    rcParams["font.sans-serif"] = ["DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def _L(cn: str, en: str) -> str:
    """根据是否有中文字体选择标签。"""
    return cn if _USE_CN else en

COMPARE_DIR = Path("dataset/compare")
INDEX_JSON = COMPARE_DIR / "index.json"
MODEL_CKPT = Path("checkpoints/wgan.pt")
MODEL_JSON = COMPARE_DIR / "model.json"


def points_to_arrays(points):
    """points [{x,y,t}, ...] -> t, x, y 数组。"""
    t = np.array([p["t"] for p in points], dtype=float)
    x = np.array([p["x"] for p in points], dtype=float)
    y = np.array([p["y"] for p in points], dtype=float)
    return t, x, y


def compute_velocity_and_acceleration(t, x, y):
    """由 t,x,y 计算速度 v (标量: 沿路径) 和加速度 a。"""
    if len(t) < 2:
        return np.array([0]), np.array([0]), np.array([0]), np.array([0])
    dt = np.diff(t)
    dt = np.maximum(dt, 1e-6)
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    v = ds / dt
    t_mid = (t[:-1] + t[1:]) / 2
    if len(v) < 2:
        return t_mid, v, t_mid, np.zeros_like(v)
    dv = np.diff(v)
    dt_mid = np.diff(t_mid)
    dt_mid = np.maximum(dt_mid, 1e-6)
    a = dv / dt_mid
    t_acc = (t_mid[:-1] + t_mid[1:]) / 2
    return t_mid, v, t_acc, a


def compute_distance_and_avg_speed(t, x, y):
    """沿轨迹的累计距离 s 与到每点的平均速度 (s, v_avg)。"""
    if len(t) < 2:
        return np.array([0]), np.array([0]), t
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.cumsum(ds)
    s = np.concatenate([[0], s])
    t_cum = t - t[0]
    t_cum = np.maximum(t_cum, 1e-6)
    v_avg = s / t_cum
    return s, v_avg, t


def compute_jitter(t, x, y, window=5):
    """滑动窗口内位移的方差作为抖动。返回与 t 对齐的 jitter 序列。"""
    if len(t) < 2:
        return np.array([0]), np.array([0])
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx**2 + dy**2)
    n = len(ds)
    jitter = np.zeros(n)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        jitter[i] = np.var(ds[lo:hi]) if hi > lo else 0
    t_j = (t[:-1] + t[1:]) / 2
    return t_j, jitter


def catmull_rom_spline_2d(x, y, samples_per_seg=25):
    """用 Catmull-Rom 样条把 2D 点列平滑成曲线采样点。

    说明：matplotlib 线段本质上仍是“点与点相连”，但通过样条插值生成更密的点，
    视觉上是平滑曲线（不再是稀疏折线）。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n <= 2:
        return x, y
    if n == 3:
        # 3 点不足以形成稳定的 CR 分段，做简单加密插值
        t = np.linspace(0.0, 1.0, max(2, samples_per_seg), endpoint=True)
        xx = np.interp(t, [0.0, 0.5, 1.0], x)
        yy = np.interp(t, [0.0, 0.5, 1.0], y)
        return xx, yy

    # 端点复制 padding
    px = np.concatenate([[x[0]], x, [x[-1]]])
    py = np.concatenate([[y[0]], y, [y[-1]]])

    ts = np.linspace(0.0, 1.0, max(5, samples_per_seg), endpoint=False)
    xs = []
    ys = []

    # 分段：P1->P2
    for i in range(1, n):
        p0 = np.array([px[i - 1], py[i - 1]])
        p1 = np.array([px[i], py[i]])
        p2 = np.array([px[i + 1], py[i + 1]])
        p3 = np.array([px[i + 2], py[i + 2]])

        t = ts[:, None]
        t2 = t * t
        t3 = t2 * t
        pts = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
        xs.append(pts[:, 0])
        ys.append(pts[:, 1])

    # 追加最后一个点，保证收尾
    xs = np.concatenate(xs + [x[-1:]])
    ys = np.concatenate(ys + [y[-1:]])
    return xs, ys


def plot_xy_paths(pairs, out=Path("dataset/compare/trajectory_xy.png")):
    """在二维坐标系 X-Y 上绘制轨迹路径（平滑曲线）。

    展示方式：一行两图，左人工、右模型，便于对比。
    """
    if not pairs:
        return
    n = len(pairs)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 1)))

    fig, (ax_h, ax_m) = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    fig.suptitle(_L(f"二维轨迹路径对比（共 {n} 条）", f"2D Path (X-Y) Comparison ({n} traces)"), fontsize=13)

    _h = _L("人工轨迹", "Human Trajectory")
    _m = _L("模型轨迹", "Model Trajectory")

    # 统一坐标范围（用原始点，不依赖插值）
    all_x = np.concatenate([np.asarray(p["x_h"], dtype=float) for p in pairs] + [np.asarray(p["x_m"], dtype=float) for p in pairs])
    all_y = np.concatenate([np.asarray(p["y_h"], dtype=float) for p in pairs] + [np.asarray(p["y_m"], dtype=float) for p in pairs])
    xmin, xmax = float(np.min(all_x)), float(np.max(all_x))
    ymin, ymax = float(np.min(all_y)), float(np.max(all_y))
    pad_x = (xmax - xmin) * 0.05 + 1e-6
    pad_y = (ymax - ymin) * 0.05 + 1e-6
    xlim = (xmin - pad_x, xmax + pad_x)
    ylim = (ymin - pad_y, ymax + pad_y)

    for ax, title in [(ax_h, _h), (ax_m, _m)]:
        ax.set_title(title)
        ax.set_xlabel(_L("X (px)", "X (px)"))
        ax.set_ylabel(_L("Y (px)", "Y (px)"))
        ax.grid(True, alpha=0.3)
        # sharex/sharey 同时开启时，adjustable='datalim' 会触发 Matplotlib 的限制
        # 改用 adjustable='box' 以兼容 tight_layout，并保持等比例显示
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    for i, p in enumerate(pairs):
        c = colors[i % len(colors)]
        xh, yh = catmull_rom_spline_2d(p["x_h"], p["y_h"], samples_per_seg=30)
        xm, ym = catmull_rom_spline_2d(p["x_m"], p["y_m"], samples_per_seg=30)

        ax_h.plot(xh, yh, "-", color=c, linewidth=2.2, label=f"#{i+1}")
        ax_m.plot(xm, ym, "-", color=c, linewidth=2.2, label=f"#{i+1}")

        # 起终点标记（用原始点，避免插值偏差）
        ax_h.plot(p["x_h"][0], p["y_h"][0], marker="o", color=c, markersize=5)
        ax_h.plot(p["x_h"][-1], p["y_h"][-1], marker="x", color=c, markersize=6)
        ax_m.plot(p["x_m"][0], p["y_m"][0], marker="o", color=c, markersize=5)
        ax_m.plot(p["x_m"][-1], p["y_m"][-1], marker="x", color=c, markersize=6)

    # legend 放到各自子图，避免太挤
    ax_h.legend(loc="best", fontsize=8, title=_L("轨迹编号", "Trace #"))
    ax_m.legend(loc="best", fontsize=8, title=_L("轨迹编号", "Trace #"))
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"二维轨迹图已保存: {out}")


def load_human_trajectories():
    """加载一条或多条人工轨迹。支持两种格式：
    - 单条：{"targetDistance": D, "points": [...]}
    - 多条：{"trajectories": [{"targetDistance": D, "points": [...]}, ...]}
    返回：[(targetDistance, points), ...]
    """
    path = INDEX_JSON
    if not path.exists():
        raise FileNotFoundError(f"请先从采集页面导出轨迹到 {path}（可单条或多条）")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "trajectories" in data:
        return [(t["targetDistance"], t["points"]) for t in data["trajectories"]]
    if "points" in data:
        return [(data["targetDistance"], data["points"])]
    raise ValueError(f"{path} 格式无效，需包含 'points' 或 'trajectories'")


def get_model_trajectory(target_distance, force_regenerate=False):
    """优先用当前模型重新生成（含时间缩放）；无模型时再读已有 model.json。"""
    def _should_regenerate():
        if force_regenerate or not MODEL_JSON.exists():
            return True
        with open(MODEL_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "trajectories" in data:
            for t in data["trajectories"]:
                if abs(t.get("targetDistance", -1) - target_distance) < 1:
                    return False
            return True
        return abs(data.get("targetDistance", -1) - target_distance) > 1

    if MODEL_CKPT.exists() and _should_regenerate():
        G = load_generator(MODEL_CKPT)
        points = generate_trajectory(G, target_distance, seed=42)
        COMPARE_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_JSON, "w", encoding="utf-8") as f:
            json.dump({"targetDistance": target_distance, "points": points}, f, indent=2)
        return points
    if MODEL_JSON.exists():
        with open(MODEL_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "trajectories" in data:
            for t in data["trajectories"]:
                if abs(t["targetDistance"] - target_distance) < 1:
                    return t["points"]
        if "points" in data and abs(data["targetDistance"] - target_distance) < 1:
            return data["points"]
    raise FileNotFoundError(
        f"未找到模型 {MODEL_CKPT}，请先运行 train.py 训练；"
        "或手动将模型生成的 trajectory 保存为 dataset/compare/model.json"
    )


def get_model_trajectories(target_distances, force_regenerate=False):
    """批量获取多条模型轨迹；多条时统一写入 model.json 为 {"trajectories": [...]}。"""
    if not target_distances:
        return []
    if len(target_distances) == 1:
        return [get_model_trajectory(target_distances[0], force_regenerate)]

    def _cached_all():
        if not MODEL_JSON.exists():
            return False
        with open(MODEL_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "trajectories" not in data or len(data["trajectories"]) != len(target_distances):
            return False
        for t, D in zip(data["trajectories"], target_distances):
            if abs(t.get("targetDistance", -1) - D) > 1:
                return False
        return True

    if MODEL_CKPT.exists() and (force_regenerate or not _cached_all()):
        G = load_generator(MODEL_CKPT)
        trajectories = []
        for i, D in enumerate(target_distances):
            points = generate_trajectory(G, D, seed=42 + i)
            trajectories.append({"targetDistance": D, "points": points})
        COMPARE_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_JSON, "w", encoding="utf-8") as f:
            json.dump({"trajectories": trajectories}, f, indent=2)
        return [t["points"] for t in trajectories]

    if MODEL_JSON.exists():
        with open(MODEL_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "trajectories" in data:
            by_d = {t["targetDistance"]: t["points"] for t in data["trajectories"]}
            return [by_d[D] for D in target_distances]
    raise FileNotFoundError(
        f"未找到模型 {MODEL_CKPT}，请先运行 train.py 训练；"
        "或手动将模型轨迹保存为 dataset/compare/model.json（含 trajectories 数组）"
    )


def plot_comparison():
    human_list = load_human_trajectories()
    target_distances = [D for D, _ in human_list]
    model_list = get_model_trajectories(target_distances, force_regenerate=True)
    n = len(human_list)
    assert n == len(model_list), "人工轨迹与模型轨迹条数不一致"

    # 为每条轨迹计算各类曲线
    pairs = []
    for (D, human_pts), model_pts in zip(human_list, model_list):
        t_h, x_h, y_h = points_to_arrays(human_pts)
        t_m, x_m, y_m = points_to_arrays(model_pts)
        t_h_mid, v_h, t_h_a, a_h = compute_velocity_and_acceleration(t_h, x_h, y_h)
        t_m_mid, v_m, t_m_a, a_m = compute_velocity_and_acceleration(t_m, x_m, y_m)
        s_h, v_avg_h, _ = compute_distance_and_avg_speed(t_h, x_h, y_h)
        s_m, v_avg_m, _ = compute_distance_and_avg_speed(t_m, x_m, y_m)
        t_j_h, jitter_h = compute_jitter(t_h, x_h, y_h)
        t_j_m, jitter_m = compute_jitter(t_m, x_m, y_m)
        t_h_norm = (t_h - t_h[0]) / (t_h[-1] - t_h[0] + 1e-9)
        x_h_norm = (x_h - x_h[0]) / (x_h[-1] - x_h[0] + 1e-9)
        t_m_norm = (t_m - t_m[0]) / (t_m[-1] - t_m[0] + 1e-9)
        x_m_norm = (x_m - x_m[0]) / (x_m[-1] - x_m[0] + 1e-9)
        pairs.append({
            "t_h": t_h, "x_h": x_h, "y_h": y_h, "t_m": t_m, "x_m": x_m, "y_m": y_m,
            "t_h_mid": t_h_mid, "v_h": v_h, "t_m_mid": t_m_mid, "v_m": v_m,
            "t_h_a": t_h_a, "a_h": a_h, "t_m_a": t_m_a, "a_m": a_m,
            "s_h": s_h, "v_avg_h": v_avg_h, "s_m": s_m, "v_avg_m": v_avg_m,
            "t_j_h": t_j_h, "jitter_h": jitter_h, "t_j_m": t_j_m, "jitter_m": jitter_m,
            "t_h_norm": t_h_norm, "x_h_norm": x_h_norm, "t_m_norm": t_m_norm, "x_m_norm": x_m_norm,
            "a_h": a_h, "a_m": a_m,
        })

    # 新增：二维坐标系轨迹路径对比（曲线）
    plot_xy_paths(pairs)

    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 1)))
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    fig.suptitle(_L(f"人工轨迹 vs 模型轨迹 对比（共 {n} 条）",
                     f"Human vs Model Trajectory Comparison ({n} traces)"), fontsize=13)

    def add_curves(ax, plot_fn, x_label=None, y_label=None, title=None):
        for i, p in enumerate(pairs):
            c = colors[i % len(colors)]
            plot_fn(ax, p, i, c)
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    _h = _L("人工", "Human")
    _m = _L("模型", "Model")

    # 1. 位移-时间 X vs T
    def plot_xt(ax, p, i, c):
        ax.plot(p["t_h"], p["x_h"], "-", color=c, linewidth=2, label=f"#{i+1} {_h}")
        ax.plot(p["t_m"], p["x_m"], "--", color=c, linewidth=1.5, label=f"#{i+1} {_m}")
    add_curves(axes[0, 0], plot_xt,
               _L("时间 T (ms)", "Time T (ms)"),
               _L("位移 X (px)", "Displacement X (px)"),
               _L("位移-时间 (X vs T)", "Displacement vs Time"))

    # 2. 速度-时间
    def plot_vt(ax, p, i, c):
        ax.plot(p["t_h_mid"], p["v_h"], "-", color=c, linewidth=2, label=f"#{i+1} {_h}")
        ax.plot(p["t_m_mid"], p["v_m"], "--", color=c, linewidth=1.5, label=f"#{i+1} {_m}")
    add_curves(axes[0, 1], plot_vt,
               _L("时间 T (ms)", "Time T (ms)"),
               _L("速度 (px/ms)", "Velocity (px/ms)"),
               _L("速度-时间", "Velocity vs Time"))

    # 3. 加速度-时间
    def plot_at(ax, p, i, c):
        ax.plot(p["t_h_a"], p["a_h"], "-", color=c, linewidth=2, label=f"#{i+1} {_h}")
        ax.plot(p["t_m_a"], p["a_m"], "--", color=c, linewidth=1.5, label=f"#{i+1} {_m}")
    add_curves(axes[0, 2], plot_at,
               _L("时间 T (ms)", "Time T (ms)"),
               _L("加速度", "Acceleration"),
               _L("加速度-时间", "Acceleration vs Time"))

    # 4. 归一化形态
    def plot_norm(ax, p, i, c):
        ax.plot(p["t_h_norm"], p["x_h_norm"], "-", color=c, linewidth=2, label=f"#{i+1} {_h}")
        ax.plot(p["t_m_norm"], p["x_m_norm"], "--", color=c, linewidth=1.5, label=f"#{i+1} {_m}")
    add_curves(axes[0, 3], plot_norm,
               _L("归一化时间 (0~1)", "Norm. Time (0~1)"),
               _L("归一化位移 (0~1)", "Norm. Disp. (0~1)"),
               _L("形态对比：归一化", "Shape: Normalized"))

    # 5. 平均速度 vs 轨迹长度
    def plot_avgv(ax, p, i, c):
        ax.plot(p["s_h"], p["v_avg_h"], "-", color=c, linewidth=2, label=f"#{i+1} {_h}")
        ax.plot(p["s_m"], p["v_avg_m"], "--", color=c, linewidth=1.5, label=f"#{i+1} {_m}")
    add_curves(axes[1, 0], plot_avgv,
               _L("轨迹长度 (px)", "Path Length (px)"),
               _L("平均速度 (px/ms)", "Avg Speed (px/ms)"),
               _L("平均速度 vs 轨迹长度", "Avg Speed vs Path Length"))

    # 6. 抖动
    def plot_jitter(ax, p, i, c):
        ax.plot(p["t_j_h"], p["jitter_h"], "-", color=c, linewidth=2, label=f"#{i+1} {_h}")
        ax.plot(p["t_j_m"], p["jitter_m"], "--", color=c, linewidth=1.5, label=f"#{i+1} {_m}")
    add_curves(axes[1, 1], plot_jitter,
               _L("时间 T (ms)", "Time T (ms)"),
               _L("抖动 (位移方差)", "Jitter (disp. var.)"),
               _L("抖动/方差", "Jitter vs Time"))

    # 7. 加速度分布直方图（多条叠加）
    ax = axes[1, 2]
    for i, p in enumerate(pairs):
        c = colors[i % len(colors)]
        ax.hist(p["a_h"], bins=20, alpha=0.4, label=f"#{i+1} {_h}", color=c, density=True, histtype="step", linewidth=2)
        ax.hist(p["a_m"], bins=20, alpha=0.4, label=f"#{i+1} {_m}", color=c, density=True, histtype="step", linestyle="--")
    ax.set_xlabel(_L("加速度", "Acceleration"))
    ax.set_ylabel(_L("密度", "Density"))
    ax.set_title(_L("加速度分布", "Acceleration Distribution"))
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    axes[1, 3].set_visible(False)

    plt.tight_layout()
    out = Path("dataset/compare/comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"对比图已保存: {out}（共 {n} 条轨迹）")


def main():
    print("加载 compare 轨迹并生成模型轨迹，绘制对比图…")
    plot_comparison()


if __name__ == "__main__":
    main()
