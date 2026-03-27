# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:44:09 2023

@author: Wanxiang Shen
"""


import numpy as np
import pandas as pd
import gseapy as gp
import os
from sklearn.decomposition import PCA



class ssGSEA:
    
    def __init__(self, geneset, geneset_name):
        self.gene_sets = {geneset_name:geneset}
        self.geneset = geneset
        self.geneset_name = geneset_name        
        self.min_size = 0
        self.max_size = 1000000
        
    def fit(self, df_tpm):
        '''
        dfx: each column is one gene, each row is one samples
        '''
        dfx = df_tpm.T
        ssgsea_results = gp.ssgsea(data=dfx, 
                                   gene_sets=self.gene_sets,                                    
                                   min_size = self.min_size, 
                                   max_size = self.max_size)

        dfres = ssgsea_results.res2d
        scale_factor = (dfres.ES / dfres.NES).mean()
        self.scale_factor = scale_factor
        return self
    
    
    
    def transform(self, df_tpm):
        dfx = df_tpm.T
        ssgsea_results = gp.ssgsea(data=dfx, 
                                   gene_sets=self.gene_sets,                                    
                                   min_size = self.min_size, 
                                   max_size = self.max_size)

        dfres = ssgsea_results.res2d.set_index('Name')
        dfres['NES'] = dfres['ES'] / self.scale_factor
        dfres = dfres.loc[df_tpm.index]
        dfres['NES_%s' % self.geneset_name] = dfres['NES']
        
        return dfres[['NES_%s' % self.geneset_name]]

    
    def fit_transform(self, df_tpm):
        self.fit(df_tpm)
        return self.transform(df_tpm)
    


class avgAbundance:
    
    def __init__(self, geneset, geneset_name):
        self.gene_sets = {geneset_name:geneset}
        self.geneset = geneset
        self.geneset_name = geneset_name
        
        
    def fit(self, df_tpm):
        '''
        dfx: each column is one gene, each row is one samples
        '''
        dfx = np.log2(df_tpm + 1)
        self.used_cols = list(set(dfx.columns) & set(self.geneset))
        dfx_used = dfx[self.used_cols]        
        score = dfx_used.mean(axis=1)
        self.scale_factor = score.mean()
        return self
    
    def transform(self, df_tpm):
        dfx = np.log2(df_tpm + 1)
        dfx_used = dfx[self.used_cols]  
        score = dfx_used.mean(axis=1)
        score = score / self.scale_factor
        score.name = 'NAG_%s' % self.geneset_name
        return score.to_frame()

    
    def fit_transform(self, df_tpm):
        self.fit(df_tpm)
        return self.transform(df_tpm)



class pcaAbundance:

    def __init__(self, geneset, geneset_name, random_state=42):
        self.gene_sets = {geneset_name:geneset}
        self.geneset = geneset
        self.geneset_name = geneset_name
        self.pca = PCA(n_components=1, random_state=random_state)
        
    def fit(self, df_tpm):
        '''
        dfx: each column is one gene, each row is one samples
        '''
        dfx = np.log2(df_tpm + 1)
        self.used_cols = list(set(dfx.columns) & set(self.geneset))
        dfx_used = dfx[self.used_cols]    

        if dfx_used.shape[1] == 0:
            return
            
        self.pca.fit(dfx_used)
        return self
    
    def transform(self, df_tpm):
        columns = ['PCA_%s' % self.geneset_name]
        dfx = np.log2(df_tpm + 1)
        dfx_used = dfx[self.used_cols]  
        
        if dfx_used.shape[1] == 0:
            score = []
        else:
            score = self.pca.transform(dfx_used)
        score = pd.DataFrame(score, index = df_tpm.index, columns = columns)
        return score

    
    def fit_transform(self, df_tpm):
        self.fit(df_tpm)
        return self.transform(df_tpm)


class origAbundance:
    
    def __init__(self, geneset, geneset_name):
        self.gene_sets = {geneset_name:geneset}
        self.geneset = geneset
        self.geneset_name = geneset_name
        
        
    def fit(self, df_tpm):
        '''
        dfx: each column is one gene, each row is one samples
        '''
        dfx = np.log2(df_tpm + 1)
        self.used_cols = list(set(dfx.columns) & set(self.geneset))
        dfx_used = dfx[self.used_cols]        
        return self
    
    def transform(self, df_tpm):
        dfx = np.log2(df_tpm + 1)
        dfx_used = dfx[self.used_cols]  
        dfx_used.index.name = '%s' % self.geneset_name
        return dfx_used

    
    def fit_transform(self, df_tpm):
        self.fit(df_tpm)
        return self.transform(df_tpm)


