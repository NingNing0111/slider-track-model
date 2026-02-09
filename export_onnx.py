"""
导出 Generator 为 ONNX。

说明：
- 当前 Generator 内部含有“逐步噪声”（torch.randn），ONNX 图不支持随机算子。
  因此导出时默认关闭 step_noise（将 step_noise_alpha 置 0），随机性主要由输入 z 提供。
- 输入:
    z:    (B, 64) float32
    cond: (B, 1)  float32，取值范围 [0,1]（target_distance / D_MAX）
- 输出:
    seq:  (B, SEQ_LEN, 2) float32（归一化空间的 dx/dy）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset.preprocess import SEQ_LEN
from model.wgan import Generator


LATENT_DIM = 64


def load_generator(checkpoint_path: str, device: torch.device) -> Generator:
    G = Generator(max_len=SEQ_LEN).to(device)
    ckpt = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    if "G" in ckpt:
        G.load_state_dict(ckpt["G"])
    else:
        G.load_state_dict(ckpt)
    G.eval()
    return G


def export_onnx(
    checkpoint: str,
    out_path: str,
    opset: int = 17,
    device: str = "cpu",
    disable_step_noise: bool = True,
    dynamic_batch: bool = False,
    verify: bool = True,
) -> Path:
    dev = torch.device(device)
    G = load_generator(checkpoint, dev)

    # ONNX 不支持随机算子：导出时默认关闭逐步噪声
    if disable_step_noise:
        G.step_noise_alpha = 0.0

    z = torch.randn(1, LATENT_DIM, device=dev, dtype=torch.float32)
    cond = torch.tensor([[0.5]], device=dev, dtype=torch.float32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = None
    if dynamic_batch:
        # 注意：LSTM 在 legacy exporter 下对动态 batch 可能不稳定；
        # 业务通常 batch=1，默认不启用动态 batch。
        dynamic_axes = {
            "z": {0: "batch"},
            "cond": {0: "batch"},
            "seq": {0: "batch"},
        }

    torch.onnx.export(
        G,
        (z, cond),
        str(out_path),
        # torch>=2.10 默认 dynamo=True 会依赖 onnxscript；
        # 这里用 legacy exporter 以减少额外依赖
        dynamo=False,
        external_data=False,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["z", "cond"],
        output_names=["seq"],
        dynamic_axes=dynamic_axes,
    )
    print(f"[OK] 导出完成: {out_path}")

    if verify:
        try:
            import onnxruntime as ort
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] 未安装 onnxruntime，跳过校验: {e}")
            return out_path

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])

        # 固定输入对齐比较
        z_np = np.random.randn(1, LATENT_DIM).astype(np.float32)
        c_np = np.random.rand(1, 1).astype(np.float32)

        with torch.no_grad():
            pt = G(torch.from_numpy(z_np).to(dev), torch.from_numpy(c_np).to(dev)).cpu().numpy()

        onnx_out = sess.run(["seq"], {"z": z_np, "cond": c_np})[0]

        max_abs = float(np.max(np.abs(pt - onnx_out)))
        mean_abs = float(np.mean(np.abs(pt - onnx_out)))
        print(f"[VERIFY] abs diff: max={max_abs:.6g}, mean={mean_abs:.6g}")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Generator to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/wgan.pt")
    parser.add_argument("--out", type=str, default="checkpoints/wgan.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", type=str, default="cpu", help="cpu 或 cuda（若可用）")
    parser.add_argument("--dynamic-batch", action="store_true", help="导出支持动态 batch（不推荐）")
    parser.add_argument(
        "--keep-step-noise",
        action="store_true",
        help="保留逐步噪声（不推荐，ONNX 导出会失败或产生不确定图）",
    )
    parser.add_argument("--no-verify", action="store_true", help="跳过 onnxruntime 数值校验")
    args = parser.parse_args()

    export_onnx(
        checkpoint=args.checkpoint,
        out_path=args.out,
        opset=args.opset,
        device=args.device,
        disable_step_noise=(not args.keep_step_noise),
        dynamic_batch=args.dynamic_batch,
        verify=(not args.no_verify),
    )


if __name__ == "__main__":
    main()

