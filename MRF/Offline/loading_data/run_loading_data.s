#!/bin/bash
#
#SBATCH --job-name=load
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --mem=100GB

python loading_data.py