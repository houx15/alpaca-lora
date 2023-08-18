#!/bin/bash
#SBATCH --job-name=prompt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --constraint=a100
#SBATCH --mail-user=yh6580@princeton.edu

module purge
module load anaconda3/2023.3
conda activate llm

python sequence_classification.py --topic 'abortion' --base_model 'chainyo/alpaca-lora-7b' --data_path './dataset/abortion' --output_dir './finetune_output/abortion' --batch_size 32 --micro_batch_size 32 --num_epochs 6 --learning_rate 1e-3 --cutoff_len 256 --val_set_size 200 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --lora_target_modules '[q_proj, v_proj]' --train_on_inputs --group_by_length
