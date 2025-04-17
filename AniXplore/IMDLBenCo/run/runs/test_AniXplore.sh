base_dir="PATH_TO_OUTPUT_SOURCE/test_AniXplore/"
mkdir -p ${base_dir}
prefix="PATH_TO_DATASET_SOURCE"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=8 \
./test-AniXplore.py \
    --epoch 29 \
    --raw_img_data_root ${prefix}"/real"\
    --edited_img_data_root ${prefix}"/fake" \
    --test_data_json "./test_datasets_anime_test.json" \
    --model AniXplore \
    --world_size 1 \
    --checkpoint_path "PATH_TO_CKPTS_SROUCE/AniXplore/checkpoint-29.pth" \
    --test_batch_size 96 \
    --if_resizing \
    --image_size 512 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --if_test_PixelF1 \
    --if_test_PixelIOU \
    --if_test_ImageF1 \
    --if_test_ImageAccuracy \
2> ${base_dir}/error.log 1>${base_dir}/logs.log