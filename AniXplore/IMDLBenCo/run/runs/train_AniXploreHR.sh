export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2

base_dir="PATH_TO_OUTPUT_SOURCE/train_AniXploreHR"
mkdir -p ${base_dir}
prefix="PATH_TO_DATASET_SOURCE"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=8 \
./train-AniXplore.py \
    --raw_img_data_root ${prefix}"/real" \
    --edited_img_data_root ${prefix}"/fake" \
    --model AniXploreHR \
    --conv_pretrain True \
    --seg_pretrain_path "PATH_TO_CKPTS_SROUCE/pretrain/mit_b3.pth" \
    --world_size 1 \
    --find_unused_parameters \
    --batch_size 12 \
    --test_batch_size 32 \
    --data_path ${prefix}"/data_list/train" \
    --test_data_path ${prefix}"/data_list/test" \
    --epochs 50 \
    --lr 1e-5 \
    --image_size 1024 \
    --if_padding \
    --min_lr 5e-7 \
    --weight_decay 0.05 \
    --warmup_epochs 2 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 2 \
    --seed 40 \
    --test_period 1 \
    --num_workers 4 \
    --if_test_PixelF1 \
    --if_test_PixelIOU \
    --if_test_ImageF1 \
    --if_test_ImageAccuracy \
2> ${base_dir}/error.log 1>${base_dir}/logs.log
