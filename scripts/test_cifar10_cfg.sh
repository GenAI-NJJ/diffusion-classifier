CUDA_VISIBLE_DEVICES=1 python eval_prob_adaptive_cfg.py \
  --version 1-4 \
  --n_workers 100 \
  --worker_idx 1 \
  --img_size 256 \
  --batch_size 192 \
  --dataset cifar10 \
  --split test \
  --n_trials 1 \
  --to_keep 1 \
  --n_samples 128 \
  --loss l2 \
  --noise_path noise_1024_32.pt \
  --prompt_path prompts/cifar10_prompts_cfg.csv

CUDA_VISIBLE_DEVICES=0 python eval_prob_adaptive_cfg.py \
  --version 1-4 \
  --n_workers 100 \
  --worker_idx 1 \
  --img_size 256 \
  --batch_size 192 \
  --dataset cifar10 \
  --split test \
  --n_trials 1 \
  --to_keep 1 \
  --n_samples 32 \
  --loss l2 \
  --noise_path noise_1024_32.pt \
  --extra nti_base_w1_n32 \
  --prompt_path prompts/cifar10_prompts_nti.csv