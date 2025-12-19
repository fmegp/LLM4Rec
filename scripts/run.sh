#!/bin/sh

lambda_V=1

# Local-path workflow (Colab notebook is recommended for end-to-end runs)
dataset_dir="/path/to/extracted/dataset/beauty"
output_dir="/path/to/output/run_$(date +%Y%m%d_%H%M%S)"

python src/training.py --dataset_dir "$dataset_dir" --output_dir "$output_dir" --lambda_V "$lambda_V"
python src/finetuning.py --dataset_dir "$dataset_dir" --output_dir "$output_dir" --pretrained_dir "$output_dir/training" --lambda_V "$lambda_V"
python src/predict.py --dataset_dir "$dataset_dir" --output_dir "$output_dir" --rec_embeddings_dir "$output_dir/finetuning/rec" --lambda_V "$lambda_V"