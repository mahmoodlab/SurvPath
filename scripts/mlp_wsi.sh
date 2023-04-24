#!/bin/bash

DATA_ROOT_DIR='/path/to/features/' # where are the TCGA features stored?
BASE_DIR="/path/to/root/dir" # where is the repo cloned?
STUDY="brca" # which disease are you working with?
TYPE_OF_PATH="combine" # what type of pathways? 
MODEL="mlp_wsi" # what type of model do you want to train?

CUDA_VISIBLE_DEVICES=1 python main.py \
    --study tcga_${STUDY} --task survival --split_dir splits --which_splits 5foldcv \
    --type_of_path $TYPE_OF_PATH --modality $MODEL --data_root_dir $DATA_ROOT_DIR --label_file datasets_csv/metadata_all_endpoints/tcga_${STUDY}.csv \
    --omics_dir datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY} --results_dir "results_blca" \
    --batch_size 1 --lr 0.0005 --opt radam --reg 0.0001 \
    --alpha_surv 0.5 --weighted_sample --max_epochs 5 --encoding_dim 768 \
    --label_col survival_months_dss --k 5 --bag_loss nll_surv --n_classes 4 --num_patches 4096 --wsi_projection_dim 256