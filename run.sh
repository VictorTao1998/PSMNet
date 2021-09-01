#!/bin/bash
export PYTHONWARNINGS="ignore"

python /cephfs/jianyu/PSMNet/main.py \
    --datapath /cephfs/datasets/iccv_pnp/messy-table-dataset/v9/training \
    --trainlist /cephfs/datasets/iccv_pnp/messy-table-dataset/v9/training_lists/all_train.txt \
    --test_datapath /cephfs/datasets/iccv_pnp/messy-table-dataset/v9/training \
    --testlist /cephfs/datasets/iccv_pnp/messy-table-dataset/v9/training_lists/all_val.txt \
    --epochs 1 \
    --lrepochs "200:10" \
    --crop_width 512  \
    --crop_height 256 \
    --test_crop_width 960  \
    --test_crop_height 540 \
    --using_ns \
    --ns_size 3 \
    --model stackhourglass \
    --logdir "/cephfs/jianyu/eval/psm_eval"  \
    --batch_size 4 \
    --test_batch_size 2 \
    --summary_freq 50 \
    --test_summary_freq 500 \
    --brightness 0.5 \
    --contrast 0.5 \
    --use_blur \
    --diff_jitter \
    --kernel 3 \
    --var "0.1,2.0" \
    #--loadckpt "/cephfs/jianyu/train/cs_train/checkpoint_best.ckpt"


