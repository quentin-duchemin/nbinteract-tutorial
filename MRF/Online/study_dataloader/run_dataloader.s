#!/bin/bash
#
#SBATCH --job-name=data_loader
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=160:00:00
#SBATCH --mem=100GB

python main_study_dataloader.py