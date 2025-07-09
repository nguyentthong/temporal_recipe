torchrun --nproc_per_node=2 \
    --master_port=10069 \
    train.py \
    --cfg-path lavis/projects/malmm_v3/qa_msrvtt.yaml \
    --options \
    model.arch blip2_vicuna_instruct \
    model.model_type vicuna7b \
    model.load_finetuned False \
    model.load_pretrained True \
    model.num_query_token 32 \
    model.vit_precision fp16 \
    model.freeze_vit True \
    model.memory_bank_length 10 \
    model.num_frames 32 \
    run.init_lr 1e-4 \
    run.max_epoch 5 \
    run.num_beams 5 \
    run.batch_size_train 16 \
    run.batch_size_eval 16 \
    run.accum_grad_iters 2 \
    run.num_workers 12 \
    run.seed 42 \
    run.evaluate False \
    run.valid_splits "['test']" \
    run.report_metric True \
    run.prefix train \
    run.resume_ckpt_path ./output/internvid_qa/blip2_vicuna_instruct_vicuna7b/train/_freezevit/checkpoint_latest.pth
