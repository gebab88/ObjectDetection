#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def read_classes(path: Path) -> list[str]:
    classes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    classes = [line for line in classes if line and not line.startswith("#")]
    if not classes:
        raise ValueError(f"No classes found in {path}")
    return classes


def write_classes(path: Path, classes: list[str]) -> None:
    path.write_text("\n".join(classes) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a YOLOE ONNX model with a baked-in class vocabulary.")
    parser.add_argument("--weights", default="yoloe-11l-seg.pt",
                        help="YOLOE checkpoint to export. Ultralytics downloads it if missing.")
    parser.add_argument("--classes", type=Path, default=Path("yoloe_classes.txt"),
                        help="Text file with one target class or phrase per line.")
    parser.add_argument("--output", type=Path, default=Path("yoloe-11l-seg.onnx"),
                        help="Output ONNX file used by config.yaml.")
    parser.add_argument("--runtime-classes", type=Path, default=Path("yoloe_classes.txt"),
                        help="Class names file used by the C++ runtime.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square export size.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--nms", action="store_true",
                        help="Embed NMS into the ONNX graph. Leave off for host-side NMS in C++.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classes = read_classes(args.classes)

    try:
        from ultralytics import YOLOE
    except ImportError as exc:
        raise SystemExit("Install ultralytics first: python3 -m pip install ultralytics") from exc

    model = YOLOE(args.weights)
    model.set_classes(classes, model.get_text_pe(classes))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.runtime_classes.parent.mkdir(parents=True, exist_ok=True)

    exported = Path(model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, nms=args.nms))
    if exported.resolve() != args.output.resolve():
        shutil.move(str(exported), args.output)

    write_classes(args.runtime_classes, classes)
    print(f"Exported {args.output}")
    print(f"Wrote runtime classes to {args.runtime_classes}")
    print(f"Classes: {len(classes)}")


if __name__ == "__main__":
    main()
