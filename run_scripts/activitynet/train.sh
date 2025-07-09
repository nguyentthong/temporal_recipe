torchrun --nproc_per_node=4 \
    --master_port=34651 \
    train.py \
    --cfg-path lavis/projects/malmm/qa_activitynet.yaml \
    --options \
    model.arch blip2_vicuna_instruct \
    model.model_type vicuna7b \
    model.load_finetuned False \
    model.load_pretrained True \
    model.num_query_token 32 \
    model.vit_precision fp16 \
    model.freeze_vit True \
    model.memory_bank_length 10 \
    model.num_frames 20 \
    run.init_lr 1e-4 \
    run.max_epoch 5 \
    run.num_beams 5 \
    run.batch_size_train 32 \
    run.batch_size_eval 32 \
    run.accum_grad_iters 1 \
    run.num_workers 12 \
    run.seed 42 \
    run.evaluate False \
    run.valid_splits "['val', 'test']" \
    run.report_metric True \
    run.prefix train
    # run.resume_ckpt_path null

