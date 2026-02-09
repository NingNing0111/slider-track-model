"""
一个简单的滑块拖拽 GUI：
1) 输入起点坐标 (x, y) 和向右滑动距离 (px)
2) 生成“类人”轨迹（优先使用本仓库的 WGAN 模型；也可用规则轨迹兜底）
3) 控制鼠标执行：起点按下 ~1s -> 按轨迹移动 -> 松开

安全措施：
- 执行前 3 秒倒计时
- 支持 ESC 紧急停止（会尽量松开鼠标）
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import tkinter as tk
from tkinter import messagebox, ttk

try:
    from pynput import keyboard, mouse
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency 'pynput'.\n"
        "- If you use uv: run `uv sync`, then start with `uv run python gui_slider.py`\n"
        "- Or install into current Python: `pip install pynput`"
    ) from e


@dataclass(frozen=True)
class Point:
    """轨迹点：相对起点的位移 x/y（像素）与时间戳 t（毫秒）。"""

    x: float
    y: float
    t: float


class ToolTip:
    """简单悬浮提示（ttk/tk 通用）。"""

    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 600) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = int(delay_ms)
        self._after_id: Optional[str] = None
        self._tip: Optional[tk.Toplevel] = None

        widget.bind("<Enter>", self._on_enter, add=True)
        widget.bind("<Leave>", self._on_leave, add=True)
        widget.bind("<ButtonPress>", self._on_leave, add=True)

    def _on_enter(self, _evt=None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _on_leave(self, _evt=None) -> None:
        self._cancel()
        self._hide()

    def _cancel(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self) -> None:
        if self._tip is not None:
            return
        try:
            x = self.widget.winfo_rootx() + 10
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        except Exception:
            x, y = 100, 100

        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        tip.attributes("-topmost", True)

        # 深色小气泡
        frm = tk.Frame(tip, bg="#0b1220", bd=1, relief="solid")
        frm.pack(fill="both", expand=True)
        lbl = tk.Label(
            frm,
            text=self.text,
            bg="#0b1220",
            fg="#e2e8f0",
            justify="left",
            wraplength=420,
            padx=10,
            pady=8,
        )
        lbl.pack()
        self._tip = tip

    def _hide(self) -> None:
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


def _estimate_duration_ms(distance_px: float) -> float:
    # 与 inference.py 的经验公式保持一致（更像真实手势）
    return max(400.0, 600.0 + 4.5 * min(float(distance_px), 400.0))


def _colored_noise(n: int, std: float, alpha: float = 0.85, seed: Optional[int] = None) -> List[float]:
    """AR(1) 相关噪声，模拟手部微抖。"""
    if n <= 0 or std <= 0:
        return [0.0] * max(0, n)
    import random

    rnd = random.Random(seed)
    eps = [rnd.gauss(0.0, 1.0) for _ in range(n)]
    x = [0.0] * n
    for i in range(1, n):
        x[i] = alpha * x[i - 1] + eps[i]
    # 归一化到指定 std
    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x) / max(1, n - 1)
    s = math.sqrt(var) if var > 1e-12 else 0.0
    if s > 1e-9:
        x = [(v - mean) / s * std for v in x]
    else:
        x = [0.0] * n
    return x


def generate_procedural_trajectory(
    distance_px: float,
    seed: Optional[int] = None,
    duration_ms: Optional[float] = None,
    points: Optional[int] = None,
    jitter_std_px: float = 0.8,
    jitter_alpha: float = 0.85,
    overshoot_px: float = 0.0,
    backtrack_px: float = 0.0,
) -> List[Point]:
    """
    规则轨迹：ease-out 位移曲线 + 相关抖动 + 可选“过冲/回拉”。
    输出与模型一致：相对位移 (x,y,t)。
    """
    import random

    rnd = random.Random(seed)
    d = float(distance_px)
    total_ms = float(duration_ms) if duration_ms is not None else _estimate_duration_ms(d)
    n = int(points) if points is not None else int(max(60, min(180, 60 + d * 0.25)))
    n = max(30, n)

    # 时间：基本等间隔 + 轻微抖动，确保单调递增
    t = [i * (total_ms / (n - 1)) for i in range(n)]
    for i in range(1, n - 1):
        t[i] += rnd.uniform(-1.0, 1.0)
    # 单调
    for i in range(1, n):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + 0.5
    t[0] = 0.0

    # 位移：easeOutCubic
    def ease_out_cubic(u: float) -> float:
        u = max(0.0, min(1.0, u))
        return 1.0 - (1.0 - u) ** 3

    # 过冲/回拉：末段拼接一个小回弹
    d_main = d + float(overshoot_px)
    xs = [ease_out_cubic(i / (n - 1)) * d_main for i in range(n)]
    if backtrack_px > 0:
        # 最后 10% 做一次回拉
        k0 = int(n * 0.9)
        k0 = max(1, min(n - 2, k0))
        for i in range(k0, n):
            u = (i - k0) / max(1, (n - 1 - k0))
            xs[i] = xs[i] - float(backtrack_px) * (u ** 1.2)

    # 确保终点对齐目标距离
    end = xs[-1]
    if abs(end) < 1e-6:
        scale = 1.0
    else:
        scale = d / end
    xs = [v * scale for v in xs]

    # y 方向微抖（相关噪声）
    ys_noise = _colored_noise(n, std=jitter_std_px * 0.5, alpha=jitter_alpha, seed=None if seed is None else seed + 1337)
    ys = [v for v in ys_noise]

    # x 方向也加一点相关噪声，但要保证基本单调向右（避免回退过大）
    xj = _colored_noise(n, std=jitter_std_px, alpha=jitter_alpha, seed=None if seed is None else seed + 42)
    xs2 = []
    last = -1e9
    for i in range(n):
        v = xs[i] + xj[i]
        # 稍微约束单调（允许极小回退，但不让整体逆行）
        if v < last - 2.0:
            v = last - 2.0
        xs2.append(v)
        last = v
    # 再次终点对齐
    delta = d - xs2[-1]
    xs2 = [v + delta * (i / (n - 1)) for i, v in enumerate(xs2)]

    return [Point(x=float(xs2[i]), y=float(ys[i]), t=float(t[i])) for i in range(n)]


def _try_generate_model_trajectory(
    distance_px: float,
    seed: Optional[int],
    add_jitter: bool,
    jitter_std_px: float,
    jitter_alpha: float,
    checkpoint_path: Path,
) -> Optional[List[Point]]:
    """使用本仓库 inference.py 的 WGAN 生成轨迹。失败则返回 None。"""
    if not checkpoint_path.exists():
        return None
    try:
        from inference import load_generator, generate_trajectory

        G = load_generator(str(checkpoint_path))
        pts = generate_trajectory(
            G,
            float(distance_px),
            seed=seed,
            add_jitter=add_jitter,
            jitter_std_px=float(jitter_std_px),
            jitter_alpha=float(jitter_alpha),
        )
        return [Point(x=p["x"], y=p["y"], t=p["t"]) for p in pts]
    except Exception:
        return None


class SliderGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("滑块轨迹 GUI（生成 + 鼠标执行）")
        self.geometry("780x520")

        self._mouse = mouse.Controller()
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._last_points: Optional[List[Point]] = None
        self._tooltips: List[ToolTip] = []
        self._hotkey_capture_start = keyboard.Key.f8
        self._hotkey_capture_end = keyboard.Key.f9

        # 键盘监听：ESC 紧急停止
        self._kb_listener = keyboard.Listener(on_press=self._on_key_press)
        self._kb_listener.daemon = True
        self._kb_listener.start()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 6}

        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, **pad)

        left = ttk.Frame(frm)
        left.pack(side="left", fill="y")

        right = ttk.Frame(frm)
        right.pack(side="right", fill="both", expand=True)

        # ---- 输入区（坐标点选）----
        ttk.Label(left, text="起点坐标（屏幕像素 px）").pack(anchor="w")
        row_start = ttk.Frame(left)
        row_start.pack(fill="x", **pad)

        self.var_x1 = tk.StringVar(value="800")
        self.var_y1 = tk.StringVar(value="730")
        ttk.Label(row_start, text="X").pack(side="left")
        ent_x1 = ttk.Entry(row_start, width=8, textvariable=self.var_x1)
        ent_x1.pack(side="left", padx=4)
        ttk.Label(row_start, text="Y").pack(side="left")
        ent_y1 = ttk.Entry(row_start, width=8, textvariable=self.var_y1)
        ent_y1.pack(side="left", padx=4)
        lbl_hotkey1 = ttk.Label(row_start, text="  （鼠标放在起点上按 F8 抓取）", foreground="#334155")
        lbl_hotkey1.pack(side="left", padx=6)

        ttk.Label(left, text="终点坐标（屏幕像素 px）").pack(anchor="w")
        row_end = ttk.Frame(left)
        row_end.pack(fill="x", **pad)

        self.var_x2 = tk.StringVar(value="950")
        self.var_y2 = tk.StringVar(value="730")
        ttk.Label(row_end, text="X").pack(side="left")
        ent_x2 = ttk.Entry(row_end, width=8, textvariable=self.var_x2)
        ent_x2.pack(side="left", padx=4)
        ttk.Label(row_end, text="Y").pack(side="left")
        ent_y2 = ttk.Entry(row_end, width=8, textvariable=self.var_y2)
        ent_y2.pack(side="left", padx=4)
        lbl_hotkey2 = ttk.Label(row_end, text="  （鼠标放在终点上按 F9 抓取）", foreground="#334155")
        lbl_hotkey2.pack(side="left", padx=6)

        self._tooltips += [
            ToolTip(ent_x1, "起点 X（px）：屏幕坐标。\n执行时会先移动到起点并按下左键。"),
            ToolTip(ent_y1, "起点 Y（px）：屏幕坐标。\n通常取滑块按钮中心点。"),
            ToolTip(lbl_hotkey1, "快捷键：鼠标放到起点位置后按 F8，自动填入起点 X/Y。"),
            ToolTip(ent_x2, "终点 X（px）：屏幕坐标。\n轨迹目标位移 dx = X2 - X1（一般应为正数）。"),
            ToolTip(ent_y2, "终点 Y（px）：屏幕坐标。\n轨迹会做末端对齐，让最终 y 更接近 (Y2)。"),
            ToolTip(lbl_hotkey2, "快捷键：鼠标放到终点位置后按 F9，自动填入终点 X/Y。"),
        ]

        ttk.Separator(left).pack(fill="x", pady=10)

        # ---- 轨迹参数 ----
        ttk.Label(left, text="轨迹生成参数").pack(anchor="w")
        self.var_use_model = tk.BooleanVar(value=True)
        chk_model = ttk.Checkbutton(left, text="优先使用模型（checkpoints/wgan.pt）", variable=self.var_use_model)
        chk_model.pack(anchor="w")
        self._tooltips.append(
            ToolTip(
                chk_model,
                "勾选：优先用 WGAN 模型生成更“像人”的 dx/dy 分布。\n"
                "不勾选：使用规则轨迹（ease-out + 抖动 + 轻微过冲/回拉）。\n"
                "如果模型文件不存在或加载失败，会自动回退到规则轨迹。",
            )
        )

        row3 = ttk.Frame(left)
        row3.pack(fill="x", **pad)
        ttk.Label(row3, text="seed(可空)").pack(side="left")
        self.var_seed = tk.StringVar(value="")
        ent_seed = ttk.Entry(row3, width=10, textvariable=self.var_seed)
        ent_seed.pack(side="left", padx=6)
        self._tooltips.append(
            ToolTip(
                ent_seed,
                "随机种子：用于“复现同一条轨迹”。\n"
                "- 留空：每次生成都不同（更随机）\n"
                "- 填数字：同参数下可重复生成同轨迹",
            )
        )

        row4 = ttk.Frame(left)
        row4.pack(fill="x", **pad)
        ttk.Label(row4, text="抖动 std(px)").pack(side="left")
        self.var_jitter_std = tk.StringVar(value="0.8")
        ent_jstd = ttk.Entry(row4, width=8, textvariable=self.var_jitter_std)
        ent_jstd.pack(side="left", padx=6)
        ttk.Label(row4, text="alpha").pack(side="left")
        self.var_jitter_alpha = tk.StringVar(value="0.85")
        ent_jalpha = ttk.Entry(row4, width=6, textvariable=self.var_jitter_alpha)
        ent_jalpha.pack(side="left", padx=6)
        self._tooltips += [
            ToolTip(
                ent_jstd,
                "抖动幅度（px，标准差）。\n"
                "- 增大：轨迹更抖、更不平滑、更“人”\n"
                "- 过大：可能上下/左右偏移明显，影响落点",
            ),
            ToolTip(
                ent_jalpha,
                "抖动相关性 alpha（0~1）。\n"
                "- 越大：噪声更“连贯”，像手部慢慢漂移\n"
                "- 越小：噪声更“颗粒”，更像高频抖动\n"
                "建议范围：0.75~0.95",
            ),
        ]

        row5 = ttk.Frame(left)
        row5.pack(fill="x", **pad)
        ttk.Label(row5, text="按下时长(s)").pack(side="left")
        self.var_hold = tk.StringVar(value="1.0")
        ent_hold = ttk.Entry(row5, width=6, textvariable=self.var_hold)
        ent_hold.pack(side="left", padx=6)
        ttk.Label(row5, text="倒计时(s)").pack(side="left")
        self.var_countdown = tk.StringVar(value="3")
        ent_cd = ttk.Entry(row5, width=6, textvariable=self.var_countdown)
        ent_cd.pack(side="left", padx=6)
        self._tooltips += [
            ToolTip(
                ent_hold,
                "鼠标到起点后，先“按住不动”的时间（秒）。\n"
                "- 增大：更像人类按住后再开始拖\n"
                "- 太小：可能显得过于机械",
            ),
            ToolTip(
                ent_cd,
                "执行前倒计时（秒）。\n"
                "用于切换到目标窗口/把鼠标移开避免误点。\n"
                "设为 0 表示立即执行。",
            ),
        ]

        ttk.Separator(left).pack(fill="x", pady=10)

        # ---- 操作按钮 ----
        row_btn = ttk.Frame(left)
        row_btn.pack(fill="x", **pad)
        btn_gen = ttk.Button(row_btn, text="生成轨迹", command=self._on_generate)
        btn_gen.pack(side="left", padx=4)
        btn_exec = ttk.Button(row_btn, text="执行滑动", command=self._on_execute)
        btn_exec.pack(side="left", padx=4)
        btn_stop = ttk.Button(row_btn, text="停止(ESC)", command=self._request_stop)
        btn_stop.pack(side="left", padx=4)
        self._tooltips += [
            ToolTip(btn_gen, "生成并预览轨迹（不会移动鼠标）。"),
            ToolTip(btn_exec, "按当前参数执行鼠标拖拽。\n如果还没生成轨迹，会先自动生成一次。"),
            ToolTip(btn_stop, "请求停止执行（等价于按 ESC）。会尽量松开鼠标左键。"),
        ]

        self.lbl_status = ttk.Label(left, text="状态：就绪（按 ESC 可紧急停止）", wraplength=260)
        self.lbl_status.pack(fill="x", **pad)

        # ---- 左侧简要说明（常用调参效果）----
        help_box = ttk.LabelFrame(left, text="参数说明（简版）")
        help_box.pack(fill="x", padx=8, pady=(6, 10))
        help_text = (
            "- seed：固定可复现；留空每次不同\n"
            "- 抖动std：越大越抖（过大会跑偏）\n"
            "- alpha：越大越“连贯漂移”，越小越“颗粒抖动”\n"
            "- 按下时长：按住停顿时间，模拟人类犹豫\n"
            "- 倒计时：给你切窗口/移开鼠标的时间\n"
            "- F8：抓取当前鼠标坐标到起点X/Y\n"
            "- F9：抓取当前鼠标坐标到终点X/Y\n"
            "- ESC：紧急停止（会尽量松开左键）"
        )
        ttk.Label(help_box, text=help_text, wraplength=250, justify="left", foreground="#334155").pack(
            anchor="w", padx=8, pady=6
        )

        # ---- 预览区 ----
        ttk.Label(right, text="轨迹预览（相对位移）").pack(anchor="w", padx=8, pady=(6, 0))
        self.canvas = tk.Canvas(right, bg="#0f172a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)

        hint = (
            "提示：执行滑动会强制移动鼠标到起点并按住左键。\n"
            "执行前建议把鼠标移出目标区域，避免误点；按 ESC 可随时中断。\n"
            "起点坐标建议用：鼠标放到滑块按钮中心后按 F8 抓取。"
        )
        ttk.Label(right, text=hint, foreground="#334155", justify="left").pack(anchor="w", padx=8, pady=(0, 6))

        # ---- 右侧详细说明 ----
        detail = ttk.LabelFrame(right, text="参数解释 / 调参效果（详细）")
        detail.pack(fill="x", padx=8, pady=(0, 10))
        detail_text = (
            "起点X/Y：屏幕坐标（px）。通常取滑块按钮中心点。\n"
            "终点X/Y：屏幕坐标（px）。轨迹会根据 dx=X2-X1 生成，并尽量对齐到你选的终点。\n"
            "优先使用模型：用 WGAN 生成更接近训练数据分布的 dx/dy；失败会自动用规则轨迹兜底。\n"
            "seed：填数字可复现同一条轨迹；留空每次随机生成（更不固定）。\n"
            "抖动std：控制抖动“幅度”。越大越不平滑越像人，但过大可能上下漂移/落点不准。\n"
            "alpha：控制抖动“连续性”。越大越像手部缓慢漂移；越小越像高频抖动。\n"
            "按下时长：到起点后先按住不动的时间。一般 0.6~1.2s 更自然。\n"
            "倒计时：执行前等待，方便切到目标窗口/移开鼠标；0 表示立刻执行。"
        )
        ttk.Label(detail, text=detail_text, wraplength=460, justify="left", foreground="#334155").pack(
            anchor="w", padx=8, pady=8
        )

    def _set_status(self, text: str) -> None:
        self.lbl_status.config(text=f"状态：{text}")

    def _on_key_press(self, key) -> None:
        if key == keyboard.Key.esc:
            self._request_stop()
            return
        # 快捷键抓取当前鼠标坐标（避免鼠标离开滑块去点按钮）
        if key == self._hotkey_capture_start:
            try:
                self.after(0, self._fill_start_from_mouse)
            except Exception:
                pass
            return
        if key == self._hotkey_capture_end:
            try:
                self.after(0, self._fill_end_from_mouse)
            except Exception:
                pass

    def _request_stop(self) -> None:
        self._stop_event.set()
        self._set_status("已请求停止（将尽快松开鼠标）")

    def _fill_start_from_mouse(self) -> None:
        x, y = self._mouse.position
        self.var_x1.set(str(int(x)))
        self.var_y1.set(str(int(y)))
        self._set_status(f"已抓取起点坐标：({int(x)}, {int(y)})")

    def _fill_end_from_mouse(self) -> None:
        x, y = self._mouse.position
        self.var_x2.set(str(int(x)))
        self.var_y2.set(str(int(y)))
        self._set_status(f"已抓取终点坐标：({int(x)}, {int(y)})")

    def _parse_int(self, s: str, name: str) -> int:
        try:
            return int(float(s.strip()))
        except Exception as e:
            raise ValueError(f"{name} 需要是数字：{s!r}") from e

    def _parse_float(self, s: str, name: str) -> float:
        try:
            return float(s.strip())
        except Exception as e:
            raise ValueError(f"{name} 需要是数字：{s!r}") from e

    def _parse_seed(self) -> Optional[int]:
        s = self.var_seed.get().strip()
        if not s:
            return None
        return self._parse_int(s, "seed")

    def _generate_points(self) -> List[Point]:
        x1 = self._parse_float(self.var_x1.get(), "起点 X")
        y1 = self._parse_float(self.var_y1.get(), "起点 Y")
        x2 = self._parse_float(self.var_x2.get(), "终点 X")
        y2 = self._parse_float(self.var_y2.get(), "终点 Y")

        dx = x2 - x1
        dy_target = y2 - y1
        if abs(dx) < 1e-6:
            raise ValueError("终点X 与 起点X 不能相同（dx=0）")
        if dx < 0:
            raise ValueError("当前只支持向右拖拽：请保证 终点X > 起点X")

        seed = self._parse_seed()
        jitter_std = max(0.0, self._parse_float(self.var_jitter_std.get(), "抖动 std"))
        jitter_alpha = self._parse_float(self.var_jitter_alpha.get(), "抖动 alpha")
        jitter_alpha = max(0.0, min(0.999, jitter_alpha))

        ckpt = Path("checkpoints/wgan.pt")
        if self.var_use_model.get():
            pts = _try_generate_model_trajectory(
                distance_px=dx,
                seed=seed,
                add_jitter=True,
                jitter_std_px=jitter_std,
                jitter_alpha=jitter_alpha,
                checkpoint_path=ckpt,
            )
            if pts is not None:
                pts = self._force_end_align(pts, dx=dx, dy_target=dy_target)
                self._set_status(
                    f"已用模型生成轨迹：{len(pts)} 点，dx≈{dx:.1f}px 终点≈({pts[-1].x:.1f}, {pts[-1].y:.1f})"
                )
                return pts

        pts2 = generate_procedural_trajectory(
            distance_px=dx,
            seed=seed,
            jitter_std_px=jitter_std,
            jitter_alpha=jitter_alpha,
            overshoot_px=max(0.0, dx * 0.02),
            backtrack_px=max(0.0, dx * 0.01),
        )
        pts2 = self._force_end_align(pts2, dx=dx, dy_target=dy_target)
        self._set_status(
            f"已用规则生成轨迹：{len(pts2)} 点，dx≈{dx:.1f}px 终点≈({pts2[-1].x:.1f}, {pts2[-1].y:.1f})"
        )
        return pts2

    def _force_end_align(self, pts: List[Point], dx: float, dy_target: float) -> List[Point]:
        """把轨迹末端强制对齐到 (dx, dy_target)，保留中间形态（用线性校正分摊误差）。"""
        if not pts:
            return pts
        n = len(pts)
        end = pts[-1]
        ex = float(dx) - float(end.x)
        ey = float(dy_target) - float(end.y)
        if abs(ex) < 1e-6 and abs(ey) < 1e-6:
            return pts
        out: List[Point] = []
        for i, p in enumerate(pts):
            u = 0.0 if n <= 1 else (i / (n - 1))
            out.append(Point(x=p.x + ex * u, y=p.y + ey * u, t=p.t))
        return out

    def _draw_preview(self, pts: List[Point]) -> None:
        self.canvas.delete("all")
        if not pts:
            return
        w = max(10, int(self.canvas.winfo_width()))
        h = max(10, int(self.canvas.winfo_height()))
        # padding
        px = 30
        py = 30

        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        dx = max(1e-6, max_x - min_x)
        dy = max(1e-6, max_y - min_y)

        # 保持 x 方向更易观察：按 x 拉伸，y 适当放大
        sx = (w - 2 * px) / dx
        sy = (h - 2 * py) / dy
        sy = max(sy, sx * 0.8)  # y 不要太扁

        def to_canvas(p: Point) -> tuple[float, float]:
            cx = px + (p.x - min_x) * sx
            cy = h - py - (p.y - min_y) * sy
            return cx, cy

        # 画坐标轴
        self.canvas.create_line(px, h - py, w - px, h - py, fill="#334155")
        self.canvas.create_line(px, py, px, h - py, fill="#334155")
        self.canvas.create_text(px + 4, h - py - 10, text="start", fill="#94a3b8", anchor="w")
        self.canvas.create_text(w - px - 4, h - py - 10, text="x+", fill="#94a3b8", anchor="e")

        # 画轨迹
        coords: List[float] = []
        for p in pts:
            cx, cy = to_canvas(p)
            coords += [cx, cy]
        if len(coords) >= 4:
            self.canvas.create_line(*coords, fill="#38bdf8", width=2)
        # 起终点
        sx0, sy0 = to_canvas(pts[0])
        sx1, sy1 = to_canvas(pts[-1])
        self.canvas.create_oval(sx0 - 4, sy0 - 4, sx0 + 4, sy0 + 4, fill="#22c55e", outline="")
        self.canvas.create_oval(sx1 - 4, sy1 - 4, sx1 + 4, sy1 + 4, fill="#f97316", outline="")

        dur = pts[-1].t - pts[0].t
        self.canvas.create_text(
            px + 8,
            py + 8,
            text=f"points={len(pts)}  duration≈{dur:.0f}ms  end≈({pts[-1].x:.1f}, {pts[-1].y:.1f})",
            fill="#e2e8f0",
            anchor="nw",
        )

    def _on_generate(self) -> None:
        try:
            pts = self._generate_points()
            self._last_points = pts
            self.after(0, lambda: self._draw_preview(pts))
        except Exception as e:
            messagebox.showerror("生成失败", str(e))
            self._set_status(f"生成失败：{e}")

    def _on_execute(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showwarning("正在执行", "当前正在执行滑动，请先停止或等待完成。")
            return
        try:
            x0 = self._parse_int(self.var_x1.get(), "起点 X")
            y0 = self._parse_int(self.var_y1.get(), "起点 Y")
            hold_s = max(0.0, self._parse_float(self.var_hold.get(), "按下时长"))
            countdown = max(0, self._parse_int(self.var_countdown.get(), "倒计时"))
            pts = self._last_points or self._generate_points()
            self._last_points = pts
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return

        abs_pts = [(x0 + p.x, y0 + p.y, p.t) for p in pts]
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._drag_worker,
            args=(abs_pts, hold_s, countdown),
            daemon=True,
        )
        self._worker.start()

    def _drag_worker(self, abs_pts: List[tuple[float, float, float]], hold_s: float, countdown: int) -> None:
        def safe_release() -> None:
            try:
                self._mouse.release(mouse.Button.left)
            except Exception:
                pass

        try:
            if not abs_pts:
                self.after(0, lambda: self._set_status("无轨迹点，取消执行"))
                return

            # 倒计时
            for i in range(countdown, 0, -1):
                if self._stop_event.is_set():
                    self.after(0, lambda: self._set_status("已中断（倒计时阶段）"))
                    safe_release()
                    return
                self.after(0, lambda i=i: self._set_status(f"{i}s 后开始执行…（按 ESC 停止）"))
                time.sleep(1.0)

            x0, y0, _ = abs_pts[0]
            self._mouse.position = (int(round(x0)), int(round(y0)))
            self.after(0, lambda: self._set_status("鼠标已到起点，按下左键…"))
            self._mouse.press(mouse.Button.left)
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < hold_s:
                if self._stop_event.is_set():
                    self.after(0, lambda: self._set_status("已中断（按下阶段）"))
                    safe_release()
                    return
                time.sleep(0.01)

            self.after(0, lambda: self._set_status("开始按轨迹滑动…"))
            last_t = abs_pts[0][2]
            for (x, y, t_ms) in abs_pts[1:]:
                if self._stop_event.is_set():
                    self.after(0, lambda: self._set_status("已中断（滑动阶段）"))
                    safe_release()
                    return
                dt = max(0.0, (t_ms - last_t) / 1000.0)
                # 限制过小/过大的间隔，防止卡顿或瞬移
                dt = min(0.05, max(0.001, dt))
                time.sleep(dt)
                self._mouse.position = (int(round(x)), int(round(y)))
                last_t = t_ms

            # 松开
            self._mouse.release(mouse.Button.left)
            self.after(0, lambda: self._set_status("执行完成：已松开左键"))
        except Exception as e:
            safe_release()
            self.after(0, lambda: self._set_status(f"执行异常：{e}（已尽量松开鼠标）"))
        finally:
            safe_release()

    def _on_close(self) -> None:
        self._request_stop()
        try:
            if self._kb_listener is not None:
                self._kb_listener.stop()
        except Exception:
            pass
        self.destroy()


def main() -> None:
    app = SliderGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

