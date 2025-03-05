# pathformer + moderntcn

# 心跳分类任务
# python -u "./run t-m.py" \
python -u "./run.py" \
    --task_name classification \
    --is_training 1 \
    --num_workers 0 \
    \
    --root_path "F:/0_czl/yan2/0-1_code/2nd_s_2nd/timesNet/dataset/all_datasets/Heartbeat/" \
    --data UEA \
    --model Heartbeat \
    --model_id Heartbeat \
    --model PathFormer \
    \
    --head_dropout 0.0 \
    --dropout 0.3 \
    --class_dropout 0.1 \
    --itr 1 \
    --learning_rate 0.001 \
    --ffn_ratio 2 \
    \
    --batch_size 20 \
    --train_epochs 30 \
    --patience 5 \
    --des Exp \
    --use_multi_scale False \
    \
    --d_model 2\
    --patch_size 4 \
    --patch_stride 4 \
    --num_blocks 1 1 1 1\
    --large_size 31 29 27 19\
    --small_size 5 5 5 5\
    --dims 256 256 256 256\

    # 1. 
    # --num_blocks 1 \ 1
    # --patch_size 4 \
    # --patch_stride 4 \
    # --large_size 31 \
    # --small_size 5 \
    # --dims 256 \