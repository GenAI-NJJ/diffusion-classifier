CUDA_VISIBLE_DEVICES=0 python eval_prob_adaptive_nti.py \
  --version 1-4 \
  --n_workers 200 \
  --worker_idx 2 \
  --img_size 256 \
  --batch_size 192 \
  --dataset cifar10 \
  --split test \
  --n_trials 1 \
  --to_keep 1 \
  --n_samples 128 \
  --loss l2 \
  --noise_path noise_1024_32.pt \
  --extra nti_final_w2 \
  --prompt_path prompts/cifar10_prompts_nti.csv

CUDA_VISIBLE_DEVICES=0 python eval_prob_adaptive_nti.py \
  --version 1-4 \
  --n_workers 100 \
  --worker_idx 0 \
  --img_size 256 \
  --batch_size 192 \
  --dataset cifar10 \
  --split test \
  --n_trials 1 \
  --to_keep 0 \
  --n_samples 128 \
  --loss l2 \
  --noise_path noise_1024_32.pt \
  --extra nti_final_w0_s5_i5_g0.5 \
  --prompt_path prompts/cifar10_prompts_nti.csv

CUDA_VISIBLE_DEVICES=1 python eval_prob_adaptive_nti.py \
  --version 1-4 \
  --n_workers 100 \
  --worker_idx 1 \
  --img_size 256 \
  --batch_size 192 \
  --dataset cifar10 \
  --split test \
  --n_trials 1 \
  --to_keep 0 \
  --n_samples 128 \
  --loss l2 \
  --noise_path noise_1024_32.pt \
  --extra nti_final_w1_s5_i5_g0.5 \
  --prompt_path prompts/cifar10_prompts_nti.csv

CUDA_VISIBLE_DEVICES=2 python eval_prob_adaptive_nti.py \
  --version 1-4 \
  --n_workers 100 \
  --worker_idx 2 \
  --img_size 256 \
  --batch_size 192 \
  --dataset cifar10 \
  --split test \
  --n_trials 1 \
  --to_keep 0 \
  --n_samples 128 \
  --loss l2 \
  --noise_path noise_1024_32.pt \
  --extra nti_final_w2_s5_i5_g0.5 \
  --prompt_path prompts/cifar10_prompts_nti.csv

CUDA_VISIBLE_DEVICES=3 python eval_prob_adaptive_nti.py \
  --version 1-4 \
  --n_workers 100 \
  --worker_idx 3 \
  --img_size 256 \
  --batch_size 192 \
  --dataset cifar10 \
  --split test \
  --n_trials 1 \
  --to_keep 0 \
  --n_samples 128 \
  --loss l2 \
  --noise_path noise_1024_32.pt \
  --extra nti_final_w3_s5_i5_g0.5 \
  --prompt_path prompts/cifar10_prompts_nti.csv