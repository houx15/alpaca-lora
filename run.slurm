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

python run.py --topic 'abortion' --do_eval  --eval_model_path 'output/sequence/7b/abortion/regression'
