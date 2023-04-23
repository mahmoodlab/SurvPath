# Modeling Dense Multimodal Interactions Between Biological Pathways and Histology for Survival Prediction
[HTML](https://arxiv.org/abs/2304.06819)

***Guillaume Jaume*, Anurag Vaidya*, Richard Chen, Drew Williamson, Paul Liang, Faisal Mahmood***

* Contributed equally

**Summary:**   Integrating whole-slide images (WSIs) and bulk transcriptomics for predicting patient survival can improve our understanding of patient prognosis. However, this multimodal task is particularly challenging due to the different nature of these data: WSIs represent a very high-dimensional spatial description of a tumor, while bulk transcriptomics represent a global description of gene expression levels within that tumor. In this context, our work aims to address two key challenges: (1) how can we tokenize transcriptomics in a semantically meaningful and interpretable way?, and (2) how can we capture dense multimodal interactions between these two modalities? Specifically, we propose to learn biological pathway tokens from transcriptomics that can encode specific cellular functions. Together with histology patch tokens that encode the different morphological patterns in the WSI, we argue that they form appropriate reasoning units for downstream interpretability analyses. We propose fusing both modalities using a memory-efficient multimodal Transformer that can model interactions between pathway and histology patch tokens. Our proposed model, SurvPath, achieves state-of-the-art performance when evaluated against both unimodal and multimodal baselines on five datasets from The Cancer Genome Atlas. Our interpretability framework identifies key multimodal prognostic factors, and, as such, can provide valuable insights into the interaction between genotype and phenotype, enabling a deeper understanding of the underlying biological mechanisms at play.

<img width="976" alt="Screen Shot 2023-04-23 at 11 26 29 AM" src="https://user-images.githubusercontent.com/55669017/233848834-ad794419-714d-4239-8d30-ad909e9f8e3e.png">

## Installation Guide for Linux (using anaconda)
### Pre-requisities: 
- Linux (Tested on Ubuntu 18.04)
- NVIDIA GPU (Tested on Nvidia GeForce RTX 2080 Ti x 16) with CUDA 11.0 and cuDNN 7.5
- Python (3.8.13), h5py (2.10.0), matplotlib (3.6.3), numpy (1.21.6), opencv-python (4.5.1.48), openslide-python (1.2.0), openslide (3.4.1), pandas (1.4.2), pillow (9.0.1), PyTorch (1.6.5), scikit-learn (1.2.1), scipy (1.9.0), torchvision (0.13.1), captum (0.6.0), shap (0.41.0)

### Downloading TCGA Data and Pathways Compositions 
To download diagnostic WSIs (formatted as .svs files), molecular feature data and other clinical metadata, please refer  to the [NIH Genomic Data Commons Data Portal](https://portal.gdc.cancer.gov)and the [cBioPortal](https://www.cbioportal.org/). WSIs for each cancer type can be downloaded using the [GDC Data Transfer Tool](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/). To get the pathway compositions for 50 Hallmarks, refer to [MsigDB](https://www.gsea-msigdb.org/gsea/msigdb/human/genesets.jsp?collection=H). To get the Reactome pathway compositions, refer to [PARADIGM](http://paradigm.five3genomics.com)

## Processing Whole Slide Images 
To process Whole Slide Images (WSIs), first, the tissue regions in each biopsy slide are segmented using Otsu's Segmentation on a downsampled WSI using OpenSlide. The 256 x 256 patches without spatial overlapping are extracted from the segmented tissue regions at the desired magnification. Consequently, an SSL pretrained Swin Transformer [TransPath](https://github.com/Xiyue-Wang/TransPath) is used to encode raw image patches into 768-dim feature vectors, which we then save as .pt files for each WSI. The extracted features then serve as input (in a .pt file) to the network. 

## Transcriptomics and Pathway Compositions
We downloaded raw RNA-seq abundance data for the TCGA cohorts from the [Xena database](https://www.nature.com/articles/s41587-020-0546-8) and performed normalization in the dataset class. The raw data is included as CSV files [`datasets_csv`](https://github.com/ajv012/SurvPath/tree/main/datasets_csv/raw_rna_data/combine). Xena database was also used to access disease specific survival and associated censorhsip. Using the Reactome and MSigDB Hallmarks pathway compositions, we selected pathways that had more than 90% of transcriptomics data available. The compositions can be found at [`metadata`](https://github.com/ajv012/SurvPath/blob/main/datasets_csv/metadata/combine_signatures.csv).  

## Training-Validation Splits 
For evaluating the algorithm's performance, we  partitioned each dataset using 5-fold cross-validation (stratified by the site of histology slide collection). Splits for each cancer type are found in the [`splits`](https://github.com/ajv012/SurvPath/tree/main/splits/5foldcv) folder, which each contain **splits_{k}.csv** for k = 1 to 5. In each **splits_{k}.csv**, the first column corresponds to the TCGA Case IDs used for training, and the second column corresponds to the TCGA Case IDs used for validation. Slides from one case are not distributed across training and validation sets. Alternatively, one could define their own splits, however, the files would need to be defined in this format. The dataset loader for using these train-val splits are defined in the `return_splits` function in the `SurvivalDatasetFactory`.

## Running Experiments 
Refer to [`docs`](https://github.com/ajv012/SurvPath/tree/main/docs) folder for source files to train SurvPath and the baselines presented in the paper. Refer to the paper to find the hyperparameters required for training. 

## Issues 
- Please open new threads or report issues directly (for urgent blockers) to `avaidya@mit.edu`.
- Immediate response to minor issues may not be available.

## License and Usage 
If you find our work useful in your research, please consider citing our paper at:

```
@article{jaume2023modeling,
  title={Modeling Dense Multimodal Interactions Between Biological Pathways and Histology for Survival Prediction},
  author={Jaume, Guillaume and Vaidya, Anurag and Chen, Richard and Williamson, Drew and Liang, Paul and Mahmood, Faisal},
  journal={arXiv preprint arXiv:2304.06819},
  year={2023}
}
```
[Mahmood Lab](https://faisal.ai) - This code is made available under the GPLv3 License and is available for non-commercial academic purposes.
