from __future__ import print_function, division
from cProfile import label
import os
import pdb
from unittest import case
import pandas as pd
import dgl 
import pickle
import networkx as nx
import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.general_utils import _series_intersection


ALL_MODALITIES = ['rna_clean.csv']  

class SurvivalDatasetFactory:

    def __init__(self,
        study,
        label_file, 
        omics_dir,
        seed, 
        print_info, 
        n_bins, 
        label_col, 
        eps=1e-6,
        num_patches=4096,
        is_mcat=False,
        is_survpath=True,
        type_of_pathway="combine",
        ):
        r"""
        Initialize the factory to store metadata, survival label, and slide_ids for each case id. 

        Args:
            - study : String 
            - label_file : String 
            - omics_dir : String
            - seed : Int
            - print_info : Boolean
            - n_bins : Int
            - label_col: String
            - eps Float
            - num_patches : Int 
            - is_mcat : Boolean
            - is_survapth : Boolean 
            - type_of_pathway : String

        Returns:
            - None
        """

        #---> self
        self.study = study
        self.label_file = label_file
        self.omics_dir = omics_dir
        self.seed = seed
        self.print_info = print_info
        self.train_ids, self.val_ids  = (None, None)
        self.data_dir = None
        self.label_col = label_col
        self.n_bins = n_bins
        self.num_patches = num_patches
        self.is_mcat = is_mcat
        self.is_survpath = is_survpath
        self.type_of_path = type_of_pathway

        if self.label_col == "survival_months":
            self.survival_endpoint = "OS"
            self.censorship_var = "censorship"
        elif self.label_col == "survival_months_pfi":
            self.survival_endpoint = "PFI"
            self.censorship_var = "censorship_pfi"
        elif self.label_col == "survival_months_dss":
            self.survival_endpoint = "DSS"
            self.censorship_var = "censorship_dss"

        #---> process omics data
        self._setup_omics_data() 
        
        #---> labels, metadata, patient_df
        self._setup_metadata_and_labels(eps)

        #---> prepare for weighted sampling
        self._cls_ids_prep()

        #---> load all clinical data 
        self._load_clinical_data()

        #---> summarize
        self._summarize()

        #---> read the signature files for the correct model/ experiment
        if self.is_mcat:
            self._setup_mcat()
        elif self.is_survpath:
            self._setup_survpath()
        else:
            self.omic_names = []
            self.omic_sizes = []
       
    def _setup_mcat(self):
        r"""
        Process the signatures for the 6 functional groups required to run MCAT baseline
        
        Args:
            - self 
        
        Returns:
            - None 
        
        """
        self.signatures = pd.read_csv("./datasets_csv/metadata/signatures.csv")
        self.omic_names = []
        for col in self.signatures.columns:
            omic = self.signatures[col].dropna().unique()
            omic = sorted(_series_intersection(omic, self.all_modalities["rna"].columns))
            self.omic_names.append(omic)
        self.omic_sizes = [len(omic) for omic in self.omic_names]

    def _setup_survpath(self):

        r"""
        Process the signatures for the 331 pathways required to run SurvPath baseline. Also provides functinoality to run SurvPath with 
        MCAT functional families (use the commented out line of code to load signatures)
        
        Args:
            - self 
        
        Returns:
            - None 
        
        """

        # for running survpath with mcat signatures 
        # self.signatures = pd.read_csv("./datasets_csv/metadata/signatures.csv")
        
        # running with hallmarks, reactome, and combined signatures
        self.signatures = pd.read_csv("./datasets_csv/metadata/{}_signatures.csv".format(self.type_of_path))
        
        self.omic_names = []
        for col in self.signatures.columns:
            omic = self.signatures[col].dropna().unique()
            omic = sorted(_series_intersection(omic, self.all_modalities["rna"].columns))
            self.omic_names.append(omic)
        self.omic_sizes = [len(omic) for omic in self.omic_names]
            

    def _load_clinical_data(self):
        r"""
        Load the clinical data for the patient which has grade, stage, etc.
        
        Args:
            - self 
        
        Returns:
            - None
            
        """
        path_to_data = "./datasets_csv/clinical_data/{}_clinical.csv".format(self.study)
        self.clinical_data = pd.read_csv(path_to_data, index_col=0)
    
    def _setup_omics_data(self):
        r"""
        read the csv with the omics data
        
        Args:
            - self
        
        Returns:
            - None
        
        """
        self.all_modalities = {}
        for modality in ALL_MODALITIES:
            self.all_modalities[modality.split('_')[0]] = pd.read_csv(
                os.path.join(self.omics_dir, modality),
                engine='python',
                index_col=0
            )

    def _setup_metadata_and_labels(self, eps):
        r"""
        Process the metadata required to run the experiment. Clean the data. Set up patient dicts to store slide ids per patient.
        Get label dict.
        
        Args:
            - self
            - eps : Float 
        
        Returns:
            - None 
        
        """

        #---> read labels 
        self.label_data = pd.read_csv(self.label_file, low_memory=False)

        #---> minor clean-up of the labels 
        uncensored_df = self._clean_label_data()

        #---> create discrete labels
        self._discretize_survival_months(eps, uncensored_df)
    
        #---> get patient info, labels, and metada
        self._get_patient_dict()
        self._get_label_dict()
        self._get_patient_data()

    def _clean_label_data(self):
        r"""
        Clean the metadata. For breast, only consider the IDC subtype.
        
        Args:
            - self 
        
        Returns:
            - None
            
        """

        if "IDC" in self.label_data['oncotree_code']: # must be BRCA (and if so, use only IDCs)
            self.label_data = self.label_data[self.label_data['oncotree_code'] == 'IDC']

        self.patients_df = self.label_data.drop_duplicates(['case_id']).copy()
        uncensored_df = self.patients_df[self.patients_df[self.censorship_var] < 1]
        
        return uncensored_df

    def _discretize_survival_months(self, eps, uncensored_df):
        r"""
        This is where we convert the regression survival problem into a classification problem. We bin all survival times into 
        quartiles and assign labels to patient based on these bins.
        
        Args:
            - self
            - eps : Float 
            - uncensored_df : pd.DataFrame
        
        Returns:
            - None 
        
        """
        # cut the data into self.n_bins (4= quantiles)
        disc_labels, q_bins = pd.qcut(uncensored_df[self.label_col], q=self.n_bins, retbins=True, labels=False)
        q_bins[-1] = self.label_data[self.label_col].max() + eps
        q_bins[0] = self.label_data[self.label_col].min() - eps
        
        # assign patients to different bins according to their months' quantiles (on all data)
        # cut will choose bins so that the values of bins are evenly spaced. Each bin may have different frequncies
        disc_labels, q_bins = pd.cut(self.patients_df[self.label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        self.patients_df.insert(2, 'label', disc_labels.values.astype(int))
        self.bins = q_bins
        
    def _get_patient_data(self):
        r"""
        Final patient data is just the clinical metadata + label for the patient 
        
        Args:
            - self 
        
        Returns: 
            - None
        
        """
        patients_df = self.label_data[~self.label_data.index.duplicated(keep='first')] 
        patient_data = {'case_id': patients_df["case_id"].values, 'label': patients_df['label'].values} # only setting the final data to self
        self.patient_data = patient_data

    def _get_label_dict(self):
        r"""
        For the discretized survival times and censorship, we define labels and store their counts.
        
        Args:
            - self 
        
        Returns:
            - self 
        
        """

        label_dict = {}
        key_count = 0
        for i in range(len(self.bins)-1):
            for c in [0, 1]:
                label_dict.update({(i, c):key_count})
                key_count+=1

        for i in self.label_data.index:
            key = self.label_data.loc[i, 'label']
            self.label_data.at[i, 'disc_label'] = key
            censorship = self.label_data.loc[i, self.censorship_var]
            key = (key, int(censorship))
            self.label_data.at[i, 'label'] = label_dict[key]

        self.num_classes=len(label_dict)
        self.label_dict = label_dict

    def _get_patient_dict(self):
        r"""
        For every patient store the respective slide ids

        Args:
            - self 
        
        Returns:
            - None
        """
    
        patient_dict = {}
        temp_label_data = self.label_data.set_index('case_id')
        for patient in self.patients_df['case_id']:
            slide_ids = temp_label_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient:slide_ids})
        self.patient_dict = patient_dict
        self.label_data = self.patients_df
        self.label_data.reset_index(drop=True, inplace=True)

    def _cls_ids_prep(self):
        r"""
        Find which patient/slide belongs to which label and store the label-wise indices of patients/ slides

        Args:
            - self 
        
        Returns:
            - None

        """
        self.patient_cls_ids = [[] for i in range(self.num_classes)]   
        # Find the index of patients for different labels
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0] 

        # Find the index of slides for different labels
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.label_data['label'] == i)[0]

    def _summarize(self):
        r"""
        Summarize which type of survival you are using, number of cases and classes
        
        Args:
            - self 
        
        Returns:
            - None 
        
        """
        if self.print_info:
            print("label column: {}".format(self.label_col))
            print("number of cases {}".format(len(self.label_data)))
            print("number of classes: {}".format(self.num_classes))

    def _patient_data_prep(self):
        patients = np.unique(np.array(self.label_data['case_id'])) # get unique patients
        patient_labels = []
        
        for p in patients:
            locations = self.label_data[self.label_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.label_data['label'][locations[0]] # get patient label
            patient_labels.append(label)
        
        self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}

    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        _, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def return_splits(self, args, csv_path, fold):
        r"""
        Create the train and val splits for the fold
        
        Args:
            - self
            - args : argspace.Namespace 
            - csv_path : String 
            - fold : Int 
        
        Return: 
            - datasets : tuple 
            
        """

        assert csv_path 
        all_splits = pd.read_csv(csv_path)
        print("Defining datasets...")
        train_split, scaler = self._get_split_from_df(args, all_splits=all_splits, split_key='train', fold=fold, scaler=None)
        val_split = self._get_split_from_df(args, all_splits=all_splits, split_key='val', fold=fold, scaler=scaler)

        args.omic_sizes = args.dataset_factory.omic_sizes
        datasets = (train_split, val_split)
        
        return datasets

    def _get_scaler(self, data):
        r"""
        Define the scaler for training dataset. Use the same scaler for validation set
        
        Args:
            - self 
            - data : np.array

        Returns: 
            - scaler : MinMaxScaler
        
        """
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(data)
        return scaler
    
    def _apply_scaler(self, data, scaler):
        r"""
        Given the datatype and a predefined scaler, apply it to the data 
        
        Args:
            - self
            - data : np.array 
            - scaler : MinMaxScaler 
        
        Returns:
            - data : np.array """
        
        # find out which values are missing
        zero_mask = data == 0

        # transform data
        transformed = scaler.transform(data)
        data = transformed

        # rna -> put back in the zeros 
        data[zero_mask] = 0.
        
        return data

    def _get_split_from_df(self, args, all_splits, split_key: str='train', fold = None, scaler=None, valid_cols=None):
        r"""
        Initialize SurvivalDataset object for the correct split and after normalizing the RNAseq data 
        
        Args:
            - self 
            - args: argspace.Namespace 
            - all_splits: pd.DataFrame 
            - split_key : String 
            - fold : Int 
            - scaler : MinMaxScaler
            - valid_cols : List 

        Returns:
            - SurvivalDataset 
            - Optional: scaler (MinMaxScaler)
        
        """

        if not scaler:
            scaler = {}
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        mask = self.label_data['case_id'].isin(split.tolist())
        df_metadata_slide = args.dataset_factory.label_data.loc[mask, :].reset_index(drop=True)
        
        # select the rna, meth, mut, cnv data for this split
        omics_data_for_split = {}
        for key in args.dataset_factory.all_modalities.keys():
            
            raw_data_df = args.dataset_factory.all_modalities[key]
            mask = raw_data_df.index.isin(split.tolist())
            
            filtered_df = raw_data_df[mask]
            filtered_df = filtered_df[~filtered_df.index.duplicated()] # drop duplicate case_ids
            filtered_df["temp_index"] = filtered_df.index
            filtered_df.reset_index(inplace=True, drop=True)

            clinical_data_mask = self.clinical_data.case_id.isin(split.tolist())
            clinical_data_for_split = self.clinical_data[clinical_data_mask]
            clinical_data_for_split = clinical_data_for_split.set_index("case_id")
            clinical_data_for_split = clinical_data_for_split.replace(np.nan, "N/A")

            # from metadata drop any cases that are not in filtered_df
            mask = [True if item in list(filtered_df["temp_index"]) else False for item in df_metadata_slide.case_id]
            df_metadata_slide = df_metadata_slide[mask]
            df_metadata_slide.reset_index(inplace=True, drop=True)

            mask = [True if item in list(filtered_df["temp_index"]) else False for item in clinical_data_for_split.index]
            clinical_data_for_split = clinical_data_for_split[mask]
            clinical_data_for_split = clinical_data_for_split[~clinical_data_for_split.index.duplicated(keep='first')]
            

            # normalize your df 
            filtered_normed_df = None
            if split_key in ["val"]:
                
                # store the case_ids -> create a new df without case_ids
                case_ids = filtered_df["temp_index"]
                df_for_norm = filtered_df.drop(labels="temp_index", axis=1)

                # store original num_patients and num_feats 
                num_patients = df_for_norm.shape[0]
                num_feats = df_for_norm.shape[1]
                columns = {}
                for i in range(num_feats):
                    columns[i] = df_for_norm.columns[i]
                
                # flatten the df into 1D array (make it a column vector)
                flat_df = np.expand_dims(df_for_norm.values.flatten(), 1)

                # get scaler
                scaler_for_data = scaler[key]

                # normalize 
                normed_flat_df = self._apply_scaler(data = flat_df, scaler = scaler_for_data)

                # change 1D to 2D
                filtered_normed_df = pd.DataFrame(normed_flat_df.reshape([num_patients, num_feats]))

                # add in case_ids
                filtered_normed_df["temp_index"] = case_ids
                filtered_normed_df.rename(columns=columns, inplace=True)

            elif split_key == "train":
                
                # store the case_ids -> create a new df without case_ids
                
                case_ids = filtered_df["temp_index"]
                df_for_norm = filtered_df.drop(labels="temp_index", axis=1)

                # store original num_patients and num_feats 
                num_patients = df_for_norm.shape[0]
                num_feats = df_for_norm.shape[1]
                columns = {}
                for i in range(num_feats):
                    columns[i] = df_for_norm.columns[i]
                
                # flatten the df into 1D array (make it a column vector)
                flat_df = df_for_norm.values.flatten().reshape(-1, 1)
                
                # get scaler
                scaler_for_data = self._get_scaler(flat_df)

                # normalize 
                normed_flat_df = self._apply_scaler(data = flat_df, scaler = scaler_for_data)

                # change 1D to 2D
                filtered_normed_df = pd.DataFrame(normed_flat_df.reshape([num_patients, num_feats]))

                # add in case_ids
                filtered_normed_df["temp_index"] = case_ids
                filtered_normed_df.rename(columns=columns, inplace=True)

                # store scaler
                scaler[key] = scaler_for_data
                
            omics_data_for_split[key] = filtered_normed_df

        if split_key == "train":
            sample=True
        elif split_key == "val":
            sample=False
            
        split_dataset = SurvivalDataset(
            split_key=split_key,
            fold=fold,
            study_name=args.study,
            modality=args.modality,
            patient_dict=args.dataset_factory.patient_dict,
            metadata=df_metadata_slide,
            omics_data_dict=omics_data_for_split,
            data_dir=args.data_root_dir,  # os.path.join(args.data_root_dir, "{}_20x_features".format(args.combined_study)),
            num_classes=self.num_classes,
            label_col = self.label_col,
            censorship_var = self.censorship_var,
            valid_cols = valid_cols,
            is_training=split_key=='train',
            clinical_data = clinical_data_for_split,
            num_patches = self.num_patches,
            omic_names = self.omic_names,
            sample=sample
            )

        if split_key == "train":
            return split_dataset, scaler
        else:
            return split_dataset
    
    def __len__(self):
        return len(self.label_data)
    

class SurvivalDataset(Dataset):

    def __init__(self,
        split_key,
        fold,
        study_name,
        modality,
        patient_dict,
        metadata, 
        omics_data_dict,
        data_dir, 
        num_classes,
        label_col="survival_months_DSS",
        censorship_var = "censorship_DSS",
        valid_cols=None,
        is_training=True,
        clinical_data=-1,
        num_patches=4000,
        omic_names=None,
        sample=True,
        ): 

        super(SurvivalDataset, self).__init__()

        #---> self
        self.split_key = split_key
        self.fold = fold
        self.study_name = study_name
        self.modality = modality
        self.patient_dict = patient_dict
        self.metadata = metadata 
        self.omics_data_dict = omics_data_dict
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.censorship_var = censorship_var
        self.valid_cols = valid_cols
        self.is_training = is_training
        self.clinical_data = clinical_data
        self.num_patches = num_patches
        self.omic_names = omic_names
        self.num_pathways = len(omic_names)
        self.sample = sample

        # for weighted sampling
        self.slide_cls_id_prep()
    
    def _get_valid_cols(self):
        r"""
        Getter method for the variable self.valid_cols 
        """
        return self.valid_cols

    def slide_cls_id_prep(self):
        r"""
        For each class, find out how many slides do you have
        
        Args:
            - self 
        
        Returns: 
            - None
        
        """
        self.slide_cls_ids = [[] for _ in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.metadata['label'] == i)[0]

            
    def __getitem__(self, idx):
        r"""
        Given the modality, return the correctly transformed version of the data
        
        Args:
            - idx : Int 
        
        Returns:
            - variable, based on the modality 
        
        """
        
        label, event_time, c, slide_ids, clinical_data, case_id = self.get_data_to_return(idx)

        if self.modality in ['omics', 'snn', 'mlp_per_path']:

            df_small = self.omics_data_dict["rna"][self.omics_data_dict["rna"]["temp_index"] == case_id]
            df_small = df_small.drop(columns="temp_index")
            df_small = df_small.reindex(sorted(df_small.columns), axis=1)
            omics_tensor = torch.squeeze(torch.Tensor(df_small.values))
            
            return (torch.zeros((1,1)), omics_tensor, label, event_time, c, clinical_data)
        
        #@TODO what is the difference between tmil_abmil and transmil_wsi
        elif self.modality in ["mlp_per_path_wsi", "abmil_wsi", "abmil_wsi_pathways", "deepmisl_wsi", "deepmisl_wsi_pathways", "mlp_wsi", "transmil_wsi", "transmil_wsi_pathways"]:

            df_small = self.omics_data_dict["rna"][self.omics_data_dict["rna"]["temp_index"] == case_id]
            df_small = df_small.drop(columns="temp_index")
            df_small = df_small.reindex(sorted(df_small.columns), axis=1)
            omics_tensor = torch.squeeze(torch.Tensor(df_small.values))
            patch_features, mask = self._load_wsi_embs_from_path(self.data_dir, slide_ids)
            
            #@HACK: returning case_id, remove later
            return (patch_features, omics_tensor, label, event_time, c, clinical_data, mask)

        elif self.modality in ["coattn", "coattn_motcat"]:
            
            patch_features, mask = self._load_wsi_embs_from_path(self.data_dir, slide_ids)

            omic1 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[0]].iloc[idx])
            omic2 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[1]].iloc[idx])
            omic3 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[2]].iloc[idx])
            omic4 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[3]].iloc[idx])
            omic5 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[4]].iloc[idx])
            omic6 = torch.tensor(self.omics_data_dict["rna"][self.omic_names[5]].iloc[idx])

            return (patch_features, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c, clinical_data, mask)
        
        elif self.modality == "survpath":
            patch_features, mask = self._load_wsi_embs_from_path(self.data_dir, slide_ids)

            omic_list = []
            for i in range(self.num_pathways):
                omic_list.append(torch.tensor(self.omics_data_dict["rna"][self.omic_names[i]].iloc[idx]))
            
            return (patch_features, omic_list, label, event_time, c, clinical_data, mask)
        
        else:
            raise NotImplementedError('Model Type [%s] not implemented.' % self.modality)

    def get_data_to_return(self, idx):
        r"""
        Collect all metadata and slide data to return for this case ID 
        
        Args:
            - idx : Int 
        
        Returns: 
            - label : torch.Tensor
            - event_time : torch.Tensor
            - c : torch.Tensor
            - slide_ids : List
            - clinical_data : tuple
            - case_id : String
        
        """
        case_id = self.metadata['case_id'][idx]
        label = torch.Tensor([self.metadata['disc_label'][idx]]) # disc
        event_time = torch.Tensor([self.metadata[self.label_col][idx]])
        c = torch.Tensor([self.metadata[self.censorship_var][idx]])
        slide_ids = self.patient_dict[case_id]
        clinical_data = self.get_clinical_data(case_id)

        return label, event_time, c, slide_ids, clinical_data, case_id
    
    def _load_wsi_embs_from_path(self, data_dir, slide_ids):
        """
        Load all the patch embeddings from a list a slide IDs. 

        Args:
            - self 
            - data_dir : String 
            - slide_ids : List
        
        Returns:
            - patch_features : torch.Tensor 
            - mask : torch.Tensor

        """
        patch_features = []
        # load all slide_ids corresponding for the patient
        for slide_id in slide_ids:
            wsi_path = os.path.join(data_dir, '{}.pt'.format(slide_id.rstrip('.svs')))
            wsi_bag = torch.load(wsi_path)
            patch_features.append(wsi_bag)
        patch_features = torch.cat(patch_features, dim=0)

        if self.sample:
            max_patches = self.num_patches

            n_samples = min(patch_features.shape[0], max_patches)
            idx = np.sort(np.random.choice(patch_features.shape[0], n_samples, replace=False))
            patch_features = patch_features[idx, :]
        
            # make a mask 
            if n_samples == max_patches:
                # sampled the max num patches, so keep all of them
                mask = torch.zeros([max_patches])
            else:
                # sampled fewer than max, so zero pad and add mask
                original = patch_features.shape[0]
                how_many_to_add = max_patches - original
                zeros = torch.zeros([how_many_to_add, patch_features.shape[1]])
                patch_features = torch.concat([patch_features, zeros], dim=0)
                mask = torch.concat([torch.zeros([original]), torch.ones([how_many_to_add])])
        
        else:
            mask = torch.ones([1])

        return patch_features, mask

    def get_clinical_data(self, case_id):
        """
        Load all the patch embeddings from a list a slide IDs. 

        Args:
            - data_dir : String 
            - slide_ids : List
        
        Returns:
            - patch_features : torch.Tensor 
            - mask : torch.Tensor

        """
        try:
            stage = self.clinical_data.loc[case_id, "stage"]
        except:
            stage = "N/A"
        
        try:
            grade = self.clinical_data.loc[case_id, "grade"]
        except:
            grade = "N/A"

        try:
            subtype = self.clinical_data.loc[case_id, "subtype"]
        except:
            subtype = "N/A"
        
        clinical_data = (stage, grade, subtype)
        return clinical_data
    
    def getlabel(self, idx):
        r"""
        Use the metadata for this dataset to return the survival label for the case 
        
        Args:
            - idx : Int 
        
        Returns:
            - label : Int 
        
        """
        label = self.metadata['label'][idx]
        return label

    def __len__(self):
        return len(self.metadata) 