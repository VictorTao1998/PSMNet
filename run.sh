#!/bin/bash

python /cephfs/jianyu/PSMNet/main.py \
    --dataset messy_table \
    --test_dataset messy_table \
    --datapath /cephfs/datasets/iccv_pnp/messy-table-dataset/v9/training \
    --trainlist /cephfs/datasets/iccv_pnp/messy-table-dataset/v9/training_lists/all_train.txt \
    --test_datapath /cephfs/datasets/iccv_pnp/messy-table-dataset/v9/training \
    --testlist /cephfs/datasets/iccv_pnp/messy-table-dataset/v9/training_lists/all_val.txt \
    --epochs 5 \
    --lrepochs "200:10" \
    --crop_width 512  \
    --crop_height 256 \
    --test_crop_width 960  \
    --test_crop_height 540 \
    --ndisp "48,24" \
    --disp_inter_r "4,1" \
    --dlossw "0.5,2.0"  \
    --using_ns \
    --ns_size 3 \
    --model gwcnet-c \
    --logdir "/cephfs/jianyu/eval/cs_eval"  \
    --ndisps "48,24" \
    --disp_inter_r "4,1"  \
    --batch_size 2 \
    --mode train \
    --summary_freq 50 \
    --test_summary_freq 500 \
    --brightness 0.5 \
    --contrast 0.5 \
    --use_blur \
    --diff_jitter \
    --kernel 3 \
    --var "0.1,2.0" \
    #--loadckpt "/cephfs/jianyu/train/cs_train/checkpoint_best.ckpt"


