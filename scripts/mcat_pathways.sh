#!/bin/bash

DATA_ROOT_DIR='/media/ssd/survpath/' # where are the TCGA features stored?
BASE_DIR="/home/guillaume/Documents/multimodal/SurvPath" # where is the repo cloned?
TYPE_OF_PATH="combine" # what type of pathways? 
MODEL="coattn" # what type of model do you want to train?
FUSION="concat" # what type of fusion do you want to do?
STUDIES=("coadread")

for STUDY in ${STUDIES[@]};
do

    CUDA_VISIBLE_DEVICES=2 python main.py \
        --study tcga_${STUDY} --task survival --split_dir splits --which_splits 5foldcv \
        --type_of_path $TYPE_OF_PATH --modality $MODEL --data_root_dir $DATA_ROOT_DIR/$STUDY/uni_features/ --label_file datasets_csv/metadata/tcga_${STUDY}.csv \
        --omics_dir datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY} --results_dir results_${STUDY} \
        --batch_size 1 --lr 0.001 --opt radam --reg 0.0001 \
        --alpha_surv 0.5 --weighted_sample --max_epochs 2 --encoding_dim 1024 \
        --label_col survival_months_dss --k 5 --bag_loss nll_surv --n_classes 4 --num_patches 4096 --wsi_projection_dim 256 --fusion $FUSION

done 