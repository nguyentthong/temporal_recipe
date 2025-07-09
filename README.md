# Temporal Recipe

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2505.12605)
[![Project Page](https://img.shields.io/badge/Project%20Page-blue?style=for-the-badge&logo=snowflake&logoColor=white&labelColor=black)](https://nguyentthong.github.io/temporal_recipe/)

<img src="./static/images/temporal_recipe.png" style="width: 400px">

Temporal-Oriented Recipe for Transferring Large Vision-Language Model to Video Understanding

[Thong Nguyen](https://nguyentthong.github.io/), [Zhiyuan Hu](), [Xu Lin](), [Cong-Duy Nguyen](), [See-Kiong Ng](), [Luu Anh Tuan]()


## Training and Inference

### Step 1: Temporal-oriented post-training

Usage:

```
bash run_scripts/POST_TRAINING_DATASET/train.sh
```

- `POST_TRAINING_DATASET`: dataset we use for incorporating temporal knowledge into large vision-language model. It can be either `internvid`, `vidal`, and `internvid_vidal`.

### Step 2: Task-specific fine-tuning

Usage:
```
bash run_scripts/FINETUNING_DATASET/train_TASK.sh
```

- `FINETUNING_DATASET`: dataset to evaluate the model’s video understanding ability. Choices include `msrvtt` and `msvd`. 

- `TASK`: video understanding task, can be either `qa` or `cap`.

### Step 3: Evaluate

Usage:
```
bash run_scripts/FINETUNING_DATASET/test_TASK.sh
```

Choices for `FINETUNING_DATASET` and `TASK` are similar to those in step 2.