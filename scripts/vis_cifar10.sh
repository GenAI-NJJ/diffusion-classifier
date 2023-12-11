python vis_prompts.py \
  --version 1-4 \
  --n_workers 1000 \
  --worker_idx 17 \
  --img_size 256 \
  --batch_size 1 \
  --dataset cifar10 \
  --split test \
  --n_trials 1 \
  --to_keep 1 \
  --n_samples 5 \
  --loss l2 \
  --noise_path noise_1024_32.pt \
  --prompt_path prompts/cifar10_prompts_cfg.csv