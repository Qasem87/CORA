import argparse
import json
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms


def expand_path(p):
    """Expand ~ and resolve absolute path."""
    return str(Path(p).expanduser().resolve())


def load_cfg(cfg_path):
    """Load YAML config and resolve dataset paths."""
    cfg = yaml.safe_load(open(cfg_path))
    root = expand_path(cfg["data_root"])
    for split in ("train", "val"):
        cfg[split]["images"] = expand_path(Path(root) / cfg[split]["images"])
        cfg[split]["ann"] = expand_path(Path(root) / cfg[split]["ann"])
    return cfg


def build_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])


def build_loader(img_dir, ann_file, batch_size, num_workers, image_size):
    ds = CocoDetection(img_dir, ann_file, transform=build_transform(image_size))
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda x: tuple(zip(*x))  # keeps lists of images/anns
    )
    return ds, dl


def multi_pass_check(loader, passes=1, tag=""):
    for r in range(passes):
        n = 0
        for imgs, targets in loader:
            n += len(imgs)
        print(f"[{tag}] pass {r+1}: saw {n} samples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_paths.yaml",
                    help="Path to dataset config file")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    print("Resolved config:")
    print(json.dumps(cfg, indent=2))

    bsz = cfg["loader"]["batch_size"]
    nw = cfg["loader"]["num_workers"]
    im_size = cfg["loader"]["image_size"]

    # Build loaders
    train_ds, train_dl = build_loader(
        cfg["train"]["images"], cfg["train"]["ann"], bsz, nw, im_size)
    val_ds, val_dl = build_loader(
        cfg["val"]["images"], cfg["val"]["ann"], bsz, nw, im_size)

    print(f"\nTrain size: {len(train_ds)} | Val size: {len(val_ds)}")

    # Multi-pass validation
    multi_pass_check(train_dl, passes=2, tag="train")
    multi_pass_check(val_dl, passes=2, tag="val")

    # Show one example
    if len(train_ds) > 0:
        img, anns = train_ds[0]
        print(f"\nSample[0]: img tensor {tuple(img.shape)}, #anns={len(anns)}")


if __name__ == "__main__":
    main()
