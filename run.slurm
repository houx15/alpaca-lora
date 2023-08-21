#!/bin/bash
#SBATCH --job-name=prompt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --constraint=a100
#SBATCH --mail-user=

module purge
module load anaconda3/2023.3
conda activate Alpaca

python run.py --topic 'abortion' --do_train --strategy prompt --model_type llama-2-13b --micro_batch_size 16
