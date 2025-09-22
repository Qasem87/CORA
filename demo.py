# demo.py
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import os
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from util.visualizer import COCOVisualizer
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.coco import make_coco_transforms
from models import build_model
from main import get_args_parser

def main(args):
    utils.init_distributed_mode(args)

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only."
    print(args)

    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, post_processors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of params in model: ', n_parameters)

    # Load checkpoint if provided.
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        if 'model_ema' in checkpoint and isinstance(checkpoint['model_ema'], dict):
            print("Loading EMA weights...")
            model_without_ddp.load_state_dict(checkpoint['model_ema'])
        elif 'model' in checkpoint:
            print("EMA not available, loading regular model weights...")
            model_without_ddp.load_state_dict(checkpoint['model'])
        else:
            print("Loading checkpoint as state_dict...")
            model_without_ddp.load_state_dict(checkpoint)

    DETECTION_THRESHOLD = 0.1

    dataset_val = build_dataset(image_set='val', args=args)

    if hasattr(dataset_val, 'coco'):
        cocojs = dataset_val.coco.dataset
        id2name = {item['id']: item['name'] for item in cocojs['categories']}

    model.eval()
    vslzr = COCOVisualizer()

    def visualize(img, targets, results, filter_fn, name):
        mask = filter_fn(results)
        vslzr.visualize(img[0], dict(
            boxes=results['boxes'][mask],
            size=targets['orig_size'],
            box_label=[f"{results['box_label'][i]}_{results['scores'][i].item():.3f}" for i, p in enumerate(mask) if p],
            image_id=idx,
        ), caption=name, savedir=os.path.join(args.output_dir, "vis"), show_in_console=True)
        print("Visualization saved.")

    with torch.no_grad():
        for idx in range(2880):  # change range or add --image_idx for customization
            img, targets = dataset_val[idx]
            w, h = img.shape[-2:]
            target_sizes = torch.tensor([[w, h]], device=device)
            img = img.to(device).unsqueeze(0)

            outputs = model(img, categories=dataset_val.category_list)
            res = post_processors['bbox'](outputs, target_sizes)[0]

            # Force only the top scoring prediction (1 box, 1 label)
            if res["scores"].numel() > 0:
                top = res["scores"].argmax().item()
                res["scores"] = res["scores"][top:top+1]
                res["labels"] = res["labels"][top:top+1]
                res["boxes"] = res["boxes"][top:top+1]
                res["box_label"] = [id2name[dataset_val.label2catid[int(res['labels'][0])]]]
            else:
                res["box_label"] = []

            results = res
            def score(results):
                return results['scores'] >= DETECTION_THRESHOLD

            visualize(img, targets, results, score, 'threshold0.2')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CORA Demo", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
