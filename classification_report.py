import argparse, csv
from pathlib import Path
import torch
import numpy as np
import util.misc as utils
from datasets import build_dataset
from models import build_model
from main import get_args_parser

DETECTION_THRESHOLD = 0.1
TRACKED_CATIDS = [ 2, 3]          # 1 FSquirrel  2 mouse  3 chipmunk


def save_predictions(model,post,  dataset, csv_path):
    model.eval()
    label2cat = dataset.label2catid
    device = next(model.parameters()).device
    
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "pred_category_id", "score"])

        with torch.no_grad():
            for img, tgt in dataset:
                img_id = tgt["image_id"].item()
                h, w_ = img.shape[-2:]
                sizes = torch.tensor([[h, w_]], device=device)

                outs = model(img.unsqueeze(0).to(device), categories=dataset.category_list)
                res = post["bbox"](outs, sizes)[0]

                if res["scores"].numel() == 0:
                    w.writerow([img_id, -1, 0.0])
                    continue

                top = res["scores"].argmax().item()
                lbl = int(res["labels"][top])
                score = res["scores"][top].item()
                cat_id = label2cat[lbl] if score >= DETECTION_THRESHOLD else -1
                w.writerow([img_id, cat_id, score])


def evaluate(pred_csv, gt_dict):
    tp = {c: 0 for c in TRACKED_CATIDS}
    fp = {c: 0 for c in TRACKED_CATIDS}
    fn = {c: 0 for c in TRACKED_CATIDS}

    with open(pred_csv) as f:
        rd = csv.DictReader(f)
        for row in rd:
            img_id = int(row["image_id"])
            pred_cat = int(row["pred_category_id"])
            gt_cat = gt_dict.get(img_id, None)
            if gt_cat not in TRACKED_CATIDS:
                continue

            if pred_cat == -1:
                fn[gt_cat] += 1
            elif pred_cat == gt_cat:
                tp[gt_cat] += 1
            else:
                fp[pred_cat] += 1
                fn[gt_cat] += 1
    return tp, fp, fn


def main():
    parser = argparse.ArgumentParser("Export predictions then eval", parents=[get_args_parser()])
    args = parser.parse_args()

    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    dataset_val = build_dataset("val", args)
    if hasattr(dataset_val, 'coco'):
        cocojs = dataset_val.coco.dataset
        id2name = {item['id']: item['name'] for item in cocojs['categories']}
        name2id = {v: k for k, v in id2name.items()}
        category_list = dataset_val.category_list
        dataset_val.label2catid = {i: name2id[cat] for i, cat in enumerate(category_list)}
    label2cat = dataset_val.label2catid

    model, _, post = build_model(args)
    model.to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        # robust checkpoint loader
        state = None
        for key in ("model_ema", "model", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                print(f"loaded weights from key '{key}'")
                break
        if state is None and isinstance(ckpt, dict):
            state = ckpt                          # full file is already a state_dict
        if not isinstance(state, dict):
            raise ValueError("No usable state_dict found in checkpoint")
        model.load_state_dict(state, strict=False)

    out_dir = Path(args.output_dir or "eval_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / "predictions.csv"

    print("Saving predictions to", pred_csv)
    save_predictions(model,post, dataset_val, pred_csv)

    gt_dict = {ann["image_id"]: ann["category_id"] for ann in dataset_val.coco.dataset["annotations"]}
    id2name = {c["id"]: c["name"] for c in dataset_val.coco.dataset["categories"]}
    tp, fp, fn = evaluate(pred_csv, gt_dict)

    out_csv = out_dir / "classification_report.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "TP", "FP", "FN", "Precision", "Recall", "F1-Score"])

        total_tp = total_fp = total_fn = 0

        print("\nClassification metrics per class")
        for cid in TRACKED_CATIDS:
            correct = tp[cid]
            wrong = fn[cid]
            false_pos = fp[cid]

            total_tp += correct
            total_fn += wrong
            total_fp += false_pos

            precision = correct / (correct + false_pos) if (correct + false_pos) > 0 else 0.0
            recall = correct / (correct + wrong) if (correct + wrong) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            class_name = id2name[cid]
            print(f"{class_name:12s}  TP {correct:4d}  FP {false_pos:4d}  FN {wrong:4d}  P {precision:.3f}  R {recall:.3f}  F1 {f1:.3f}")
            writer.writerow([class_name, correct, false_pos, wrong, f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}"])

        # Overall scores
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\nOverall       TP {total_tp:4d}  FP {total_fp:4d}  FN {total_fn:4d}  P {precision:.3f}  R {recall:.3f}  F1 {f1:.3f}")
        writer.writerow(["Overall", total_tp, total_fp, total_fn, f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}"])
if __name__ == "__main__":
    main()
