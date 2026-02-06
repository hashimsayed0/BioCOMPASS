# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:05:13 2023

@author: Wanxiang Shen
"""

import os
import numpy as np
import pandas as pd
from glob import glob
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.nn.functional import normalize
from torch.distributions import Beta

        
def tolist(x):
    return x.tolist()

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class NoScaler:
    def __init__(self):
        pass
    def fit(self, dfx):
        return self
        
    def transform(self, dfx):
        return dfx

    def fit_transform(self, dfx):
        return dfx

    def inverse_transform(self, dfx):
        return dfx


class TCGAData(Dataset):
    
    def __init__(self, dfcx,  dfy, augmentor, K = 1):
        '''
        dfcx: cancer type and TPM matrix, first column is cancer type, 0-32
        dfy: task dataframe, one-hot encoding for classification task
        augmentor: MixupNomralAugmentor
        K: float
        '''

        self.feature_name = dfcx.columns[1:] #first_columns is cancer type
        self.patient_name = dfcx.index
        self.task_cols = dfy.columns
        self.task_dim = len(dfy.columns)
        
        X = torch.tensor(dfcx.values,dtype=torch.float32).clone().detach()
        self.X = X

        y_scaler = MinMaxScaler()
        y = y_scaler.fit_transform(dfy.values)
        self.y = torch.tensor(y, dtype=torch.float32).clone().detach()
        self.y_scaler = y_scaler

        self.K = K

        dist = torch.cdist(X[:,1:], X[:,1:])
        #print(dist.min())
        
        if K in [-1, 1, 0]:
            knn_value, knn_idx = dist.topk(len(X), largest=False)
        else:
            knn_value, knn_idx = dist.topk(int(len(X)*K), largest=False)
            
        self.knn_value = knn_value[:, 1:]
        self.knn_idx = knn_idx[:, 1:]
        
        self.augmentor = augmentor 


    def __len__(self):
        return len(self.X)

    
    def __getitem__(self, idx):

        ## anchor sample
        a = self.X[idx]

        topK_idx = self.knn_idx[idx]
        
        ## positive sample augmentation
        xa = self.augmentor.augment_a(a)[0]
        xp = self.augmentor.augment_p(a)[0]

        ## negative sample
        neg_idx = np.random.choice(topK_idx)
        n = self.X[neg_idx]

        ya = self.y[idx]
        yp = ya.clone()
        yn = self.y[neg_idx]

        xn = self.augmentor.augment_n(n)[0]

        x = [xa, xp, xn]
        y = [ya, yp, yn]



        
        return x, y



class GeneData(Dataset):
    def __init__(self, dfcx, clinical_features=None):
        '''
        df_tpm: TPM matrix
        clinical_features: Optional dict of clinical features from ClinicalFeatureLoader
            Keys: 'treatment', 'cell_type', 'pathway', 'auxiliary_TIDE', 'auxiliary_IPRES', 'auxiliary_phenotype'
            Values: numpy arrays of shape [N, feature_dim]
        '''
        self.feature_name = dfcx.columns[1:]
        self.patient_name = dfcx.index

        X = torch.tensor(dfcx.values,
                         dtype=torch.float32).clone().detach()
        self.X = X
        self.dfcx = dfcx

        # Store clinical features as tensors
        self.clinical_features = None
        if clinical_features is not None:
            self.clinical_features = {}
            for key, arr in clinical_features.items():
                if arr is not None:
                    self.clinical_features[key] = torch.tensor(arr, dtype=torch.float32)

    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):
        ## anchor sample
        x = self.X[idx]

        # Clinical features (optional)
        if self.clinical_features is not None:
            metadata = {}
            clinical_batch = {}
            for key, tensor in self.clinical_features.items():
                clinical_batch[key] = tensor[idx]
            # Concatenate all biomarkers for biomarker attention (115 features total)
            # cell_type (43) + pathway (42) + auxiliary_TIDE (10) + auxiliary_IPRES (4) + auxiliary_phenotype (16) = 115
            biomarkers_list = []
            for key in ['cell_type', 'pathway', 'auxiliary_TIDE', 'auxiliary_IPRES', 'auxiliary_phenotype']:
                if key in clinical_batch:
                    biomarkers_list.append(clinical_batch[key])
            if biomarkers_list:
                clinical_batch['biomarkers'] = torch.cat(biomarkers_list, dim=0)
            metadata['clinical_features'] = clinical_batch
            return x, metadata
        else:
            return x





class ITRPData(Dataset):
    def __init__(self, dfcx,  dfy, clinical_features=None):

        '''
        df_tpm: TPM matrix
        df_task: task dataframe, one-hot encoding for classification task (response label)
        clinical_features: Optional dict of clinical features from ClinicalFeatureLoader
            Keys: 'treatment', 'cell_type', 'pathway', 'auxiliary_TIDE', 'auxiliary_IPRES', 'auxiliary_phenotype'
            Values: numpy arrays of shape [N, feature_dim]
        '''

        self.feature_name = dfcx.columns[1:]
        self.patient_name = dfcx.index
        self.task_cols = dfy.columns #['NR', 'R',]
        self.task_dim = len(dfy.columns)

        X = torch.tensor(dfcx.values, dtype=torch.float32).clone().detach()
        self.X = X

        y_scaler = MinMaxScaler()
        y = y_scaler.fit_transform(dfy.values)
        self.y = torch.tensor(y, dtype=torch.float32).clone().detach()
        self.y_scaler = y_scaler

        self.responder_idx = torch.where(self.y[:, 1] == 1.)[0]
        self.nonresponder_idx = torch.where(self.y[:, 1] == 0.)[0]

        # Store clinical features as tensors
        self.clinical_features = None
        if clinical_features is not None:
            self.clinical_features = {}
            for key, arr in clinical_features.items():
                if arr is not None:
                    self.clinical_features[key] = torch.tensor(arr, dtype=torch.float32)

    
    def __len__(self):
        return len(self.X)

    
    def __getitem__(self, idx):

        ya = self.y[idx]
        #print(y)
        ## case responder
        if ya[1] == 1:
            pos_idx = np.random.choice(self.responder_idx)
            neg_idx = np.random.choice(self.nonresponder_idx)
        ## case non-responder
        else:
            pos_idx = np.random.choice(self.nonresponder_idx)
            neg_idx = np.random.choice(self.responder_idx)


        ## anchor sample
        xa = self.X[idx]
        xp = self.X[pos_idx]
        xn = self.X[neg_idx]

        x = [xa, xp, xn]

        yp = self.y[pos_idx]
        yn = self.y[neg_idx]

        y = [ya, yp, yn]

        # Metadata (optional)
        metadata = {}

        # Clinical features (optional)
        if self.clinical_features is not None:
            clinical_batch = {}
            for key, tensor in self.clinical_features.items():
                clinical_batch[key] = tensor[idx]
            # Concatenate all biomarkers for biomarker attention (115 features total)
            # cell_type (43) + pathway (42) + auxiliary_TIDE (10) + auxiliary_IPRES (4) + auxiliary_phenotype (16) = 115
            biomarkers_list = []
            for key in ['cell_type', 'pathway', 'auxiliary_TIDE', 'auxiliary_IPRES', 'auxiliary_phenotype']:
                if key in clinical_batch:
                    biomarkers_list.append(clinical_batch[key])
            if biomarkers_list:
                clinical_batch['biomarkers'] = torch.cat(biomarkers_list, dim=0)
            metadata['clinical_features'] = clinical_batch

        # Return with metadata if any exists, otherwise backward compatible
        if metadata:
            return x, y, metadata
        else:
            return x, y