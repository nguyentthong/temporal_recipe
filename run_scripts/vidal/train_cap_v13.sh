torchrun --nproc_per_node=8 \
    --master_port=11013 \
    train.py \
    --cfg-path lavis/projects/malmm_v13/cap_vidal.yaml \
    --options \
    model.arch blip2_vicuna_instruct \
    model.model_type vicuna7b \
    model.load_finetuned False \
    model.load_pretrained True \
    model.num_query_token 64 \
    model.vit_precision fp16 \
    model.freeze_vit True \
    model.memory_bank_length 40 \
    model.num_frames 32 \
    model.num_experts 0 \
    run.init_lr 1e-5 \
    run.max_epoch 1 \
    run.num_beams 5 \
    run.batch_size_train 4 \
    run.batch_size_eval 16 \
    run.accum_grad_iters 8 \
    run.num_workers 12 \
    run.seed 42 \
    run.evaluate False \
    run.valid_splits "['test']" \
    run.report_metric True \
    run.prefix train
    # run.resume_ckpt_path
