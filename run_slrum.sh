#!/bin/bash

export PATH=/mnt/lustre/share/cuda-10.2/bin/:$PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-10.2/lib64:$LD_LIBRARY_PATH

job_name="container"

#srun --partition=shlab_cv_gp  -N1 --gres=gpu:8 --cpus-per-task=8 --job-name=$job_name --kill-on-bad-exit=1 \
#     python -m torch.distributed.launch --nproc_per_node=8 --use_env  main.py \
#     --model container_v1_light \
#     --batch-size 128 \
#     --data-path s3://GCC/AdamGCC \
#     --output_dir /mnt/lustre/zhoujingqiu.vendor/output/container_clip_RN50x16

srun --partition=shlab_cv_gp  -N1 --gres=gpu:8 --cpus-per-task=8 --job-name=$job_name --kill-on-bad-exit=1 \
     python -m torch.distributed.launch --nproc_per_node=8 --use_env  main.py \
     --model container_v1_light \
     --batch-size 128 \
     --data-path s3://GCC/AdamGCC \
     --meta_file /mnt/cache/zhoujingqiu.vendor/universal_pix2seq/gcc_train_RN101_img_k1000.json\
     --output_dir /mnt/lustre/zhoujingqiu.vendor/output/container_clip_RN101_image_topk1000