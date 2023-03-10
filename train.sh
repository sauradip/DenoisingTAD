# First stage
# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=11323 --use_env main.py --window_size 100 --batch_size 128 --stage 1 --num_queries 32 --point_prob_normalize

# Second stage for relaxation mechanism
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=11324 --use_env main.py --window_size 100 --batch_size 128 --lr 1e-5 --stage 2 --epochs 10 --lr_drop 5 --num_queries 32 --point_prob_normalize --load /media/phd/SAURADIP5TB/DiffTAD/output/checkpoint_best_sum_ar.pth

# Third stage for completeness head
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=11325 --use_env main.py --window_size 100 --batch_size 128 --lr 1e-4 --stage 3 --epochs 20 --num_queries 32 --point_prob_normalize --load /media/phd/SAURADIP5TB/DiffTAD/output/checkpoint_best_sum_ar.pth
