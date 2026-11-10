#!/usr/bin/env python3
"""Export a YOLO model to an Apple Core AI asset in FP16, INT8 or INT4.

Ultralytics' Core AI export only offers FP32 and FP16 (quantize=16). For
--quantize 8 or 4 this script hooks into that export and replaces the
convolution weights with coreai-torch's weight-only quantization op
(constexpr_blockwise_shift_scale) before the model is traced. Activations stay
FP16, so the asset has the same float16 input and output as an FP16 export and
runs with the Core AI backend unchanged.

  --quantize 16  FP16 weights (same as `yolo export ... quantize=16`)
  --quantize 8   INT8 weights, one scale per output channel (W8A16)
  --quantize 4   INT4 weights, one scale per block of input channels (W4A16)

Measured with yolo26x on an M5 Pro (coco128 val; app on the 1025-frame test video):

  FP16  107 MB  mAP50-95 0.661   6.4 ms/frame
  INT8   54 MB  mAP50-95 0.660   6.7 ms/frame
  INT4   29 MB  mAP50-95 0.591  12.5 ms/frame
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a YOLO model to Core AI with FP16, INT8 or INT4 weights.")
    parser.add_argument("--weights", default="yolo26x.pt",
                        help="YOLO checkpoint to export. Ultralytics downloads it if missing.")
    parser.add_argument("--quantize", type=int, choices=(16, 8, 4), default=8,
                        help="Weight precision: 16 (FP16), 8 (INT8) or 4 (INT4).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output .aimodel asset (default: <weights stem>[_q8|_q4].aimodel). "
                             "Keep the 'yolo26' substring in the name so the model format is detected.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square export size.")
    parser.add_argument("--block-size", type=int, default=32,
                        help="INT4 only: input channels sharing one scale. Layers whose input channel "
                             "count is not a multiple fall back to one scale per output channel.")
    parser.add_argument("--min-weights", type=int, default=4096,
                        help="Convolutions with fewer weights (the stem, depthwise kernels) stay FP16.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output asset.")
    return parser.parse_args()


def make_quant_conv(nbits: int, block_size: int):
    """Build the Conv2d replacement class; imported lazily so --help works without the dependencies."""
    import torch
    import torch.nn.functional as F
    from coreai_torch._compression.custom_layers import WeightDequantizeModule

    class QuantConv2d(torch.nn.Module):
        """Conv2d whose weight is dequantized from INT8/INT4 data by a Core AI constexpr op."""

        def __init__(self, conv: torch.nn.Conv2d) -> None:
            super().__init__()
            w = conv.weight.detach().float()
            out_ch, in_ch = w.shape[:2]
            qmax = 2 ** (nbits - 1) - 1
            if nbits == 4 and block_size > 0 and in_ch % block_size == 0:
                # Symmetric, one scale per block of input channels: scale shape [O, I/block, 1, 1]
                blocks = w.reshape(out_ch, in_ch // block_size, block_size, *w.shape[2:])
                scale = blocks.abs().amax(dim=(2, 3, 4), keepdim=True).clamp(min=1e-8) / qmax
                q = torch.clamp(torch.round(blocks / scale), -qmax - 1, qmax).reshape_as(w)
                scale = scale.reshape(out_ch, in_ch // block_size, 1, 1)
            else:
                # Symmetric, one scale per output channel: scale shape [O, 1, 1, 1]
                scale = w.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8) / qmax
                q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
            self.deq = WeightDequantizeModule(
                quantized_data=q.to(torch.int8),  # int4 values are stored in int8 and packed at conversion
                scale=scale.to(torch.float16),
                input_dtype=torch.int4 if nbits == 4 else torch.int8,
            )
            bias = conv.bias
            self.bias = None if bias is None else torch.nn.Parameter(bias.detach().half(), requires_grad=False)
            self.stride, self.padding = conv.stride, conv.padding
            self.dilation, self.groups = conv.dilation, conv.groups

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.conv2d(x, self.deq(), self.bias, self.stride, self.padding, self.dilation, self.groups)

    return QuantConv2d


def quantize_convs(model, nbits: int, block_size: int, min_weights: int) -> tuple[int, int]:
    """Replace every large enough Conv2d in place; returns (quantized, kept in FP16)."""
    import torch

    quant_conv = make_quant_conv(nbits, block_size)
    quantized = kept = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if not isinstance(child, torch.nn.Conv2d):
                continue
            if child.weight.numel() >= min_weights and child.in_channels > 3:
                setattr(parent, name, quant_conv(child))
                quantized += 1
            else:
                kept += 1
    return quantized, kept


def main() -> None:
    args = parse_args()
    suffix = {16: "", 8: "_q8", 4: "_q4"}[args.quantize]
    output = args.output or Path(f"{Path(args.weights).stem}{suffix}.aimodel")
    if output.exists() and not args.force:
        raise SystemExit(f"{output} already exists; pass --force to overwrite it")

    try:
        import ultralytics.utils.export.coreai as coreai_export
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install ultralytics first: python3 -m pip install ultralytics") from exc

    if args.quantize != 16:
        torch2coreai = coreai_export.torch2coreai

        def torch2coreai_quantized(model, im, output_file, quantize=None, metadata=None, prefix=""):
            model = model.half()  # FP16 activations; the int8 quantized data keeps its dtype
            quantized, kept = quantize_convs(model, args.quantize, args.block_size, args.min_weights)
            print(f"{prefix} quantized {quantized} conv weights to INT{args.quantize}, kept {kept} in FP16")
            return torch2coreai(model, im.half(), output_file, quantize=16, metadata=metadata, prefix=prefix)

        coreai_export.torch2coreai = torch2coreai_quantized

    yolo = YOLO(args.weights)
    with tempfile.TemporaryDirectory() as tmp:
        # Ultralytics writes the asset next to the checkpoint; point it at a scratch
        # directory so an existing <stem>.aimodel next to the weights is never overwritten.
        yolo.model.pt_path = str(Path(tmp) / Path(args.weights).name)
        exported = Path(yolo.export(format="coreai", imgsz=args.imgsz, quantize=16))
        if output.exists():
            shutil.rmtree(output) if output.is_dir() else output.unlink()
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported), output)
    print(f"Exported {output}")


if __name__ == "__main__":
    main()
