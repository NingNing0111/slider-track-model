"""数据集 IO：从页面导出的 JSON 加载并转为 (condition, sequence) 格式。"""
from pathlib import Path
import json


def load_sample(obj):
    """单条样本：{ targetDistance, points: [{x,y,t}, ...] } -> (D, points)."""
    points = obj["points"]
    if not points:
        return None
    D = float(obj["targetDistance"])
    return D, points


def points_to_deltas(points):
    """points [(x,y,t), ...] -> deltas [(dx,dy,dt), ...] 长度 n-1。"""
    out = []
    for i in range(1, len(points)):
        dx = points[i]["x"] - points[i - 1]["x"]
        dy = points[i]["y"] - points[i - 1]["y"]
        dt = points[i]["t"] - points[i - 1]["t"]
        dt = max(dt, 1)  # 避免除零
        out.append((dx, dy, dt))
    return out


def load_json_file(path):
    """一个 JSON 文件可能是单条或数组，返回 [(D, points), ...]。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        samples = [load_sample(s) for s in data]
    else:
        samples = [load_sample(data)]
    return [s for s in samples if s is not None]


def load_dataset(dir_path):
    """目录下所有 .json 的样本汇总。"""
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        return []
    all_samples = []
    for p in sorted(dir_path.glob("*.json")):
        all_samples.extend(load_json_file(p))
    return all_samples
