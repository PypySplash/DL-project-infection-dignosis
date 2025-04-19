#!/bin/bash
#SBATCH --job-name=covid_project
#SBATCH --output=project_%j.out
#SBATCH --error=project_%j.err
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCh --mail-type=BEGIN,END,FAIL
#SBATCh --mail-user=u1527533@utah.edu
#SBATCH --account=kingspeak-gpu
#SBATCH --partition=kingspeak-gpu

# Load required modules
module load python/3.9
module load cuda/11.8

# Activate virtual environment
source ~/pytorch_env/bin/activate

# Step 1: Data preprocessing
echo "Running data preprocessing..."
python ~/covid_dataset/filter_metadata.py

# Step 2: Train Simple CNN
echo "Training Simple CNN..."
python ~/covid_dataset/simple_cnn.py

# Step 3: Train ResNet18
echo "Training ResNet18..."
python ~/covid_dataset/train_resnet18.py

echo "All steps completed."
