#!/bin/bash
#SBATCH --job-name=prompt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=yh6580@princeton.edu

module purge
module load anaconda3/2023.3
conda activate llm

python finetune.py --topic 'abortion' --base_model 'chainyo/alpaca-lora-7b' --data_path './json/abortion' --output_dir './output/abortion' --batch_size 32 --micro_batch_size 8 --num_epochs 6 --learning_rate 1e-4 --cutoff_len 256 --val_set_size 200 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --lora_target_modules '[q_proj, v_proj]' --train_on_inputs --group_by_length
