# Prepare Datasets

We provide instruction for preparing the Rodent dataset used in CORA experiments.

<br>

## Overview
- The dataset follows the COCO format.

- It contains 3 species classes:

    - chipmunk (target/unseen class)

    - mouse

    - fsquirrel

- Training split: 80% of mouse and fsquirrel images.

- Validation split: remaining 20% of mouse and fsquirrel, plus all chipmunk images.

<br>

## Dataset structure

```
coco/
  annotations/
    instances_train2017_base.json  # for training data
    instances_val2017_basetarget   # for validation data
 train2017/
    images for training data
 val2017/
    images for validation data
    
```
<br>

## Dataset path

Set the dataset path before launching training by exporting the environment variable:

```
export data_path=/mnt/ceph/alha0230/Rodent/coco
```
<br>

## Lunching an experiment

Run the provided config script, which sources controller.sh to handle distributed training:
```
bash configs/COCO/R50_dab_ovd_3enc_apm128_splcls0.2_relabel_noinit.sh exp_name 1 local
```

<br>

##Validation
Use this step to confirm PyTorch can load both splits, iterate multiple times, and that paths are correct.
```
python scripts/validate_coco.py --config configs/data_paths.yaml

```
