import numpy as np
import pandas as pd
from tidepy.pred import TIDE as get_tide
from .base import GeneSetScorer
from .scorer import avgAbundance, pcaAbundance, origAbundance, ssGSEA
from .markers import MARKERS, GENE_ID_MAP

# map gene name to Entrez ID
name2entrez_map = GENE_ID_MAP.set_index('gene_name')['entrezgene']

'''
http://tide.dfci.harvard.edu/download/
https://github.com/liulab-dfci/TIDEpy/tree/master

Jiang P, Gu S, Pan D, Fu J, Sahu A, Hu X, Li Z, Traugh N, Bu X, Li B, et al: Signatures of T cell dysfunction and exclusion predict cancer immunotherapy response. Nat Med 2018, 24:1550-1558.

'''


class Jiang_TIDE(GeneSetScorer):
    def __init__(self, cancer_type = 'SKCM', drug_target = 'PD1'):
        self.cancer_type = cancer_type
        self.drug_target = drug_target

        self.name = 'Jiang_TIDE'
        self.reference = 'PMID: 30127393'
        self.description = '''http://tide.dfci.harvard.edu/download/; 
                              https://github.com/liulab-dfci/TIDEpy/tree/master,
                              Signatures of T cell dysfunction and exclusion predict cancer immunotherapy response. 
                              Nat Med 2018, 24:1550-1558.'''
        self.gene_set = []
        self.gs = pd.Series()

        if cancer_type == 'SKCM':
            self.cancer = 'Melanoma'
            
        elif cancer_type in ('LUAD', 'LUSC'):
            self.cancer = 'NSCLC'
        else:
            self.cancer = 'Other'

    
    def fit(self, df_train, seed=42):
        return self

    def transform(self, df_tpm):
        return self(df_tpm)

    def __call__(self, df_tpm):

        df_tpm_input = df_tpm.T
        df_tpm_input.index = df_tpm_input.index.map(name2entrez_map)
        
        result = get_tide(df_tpm_input, 
                      cancer=self.cancer,
                      pretreat=False,
                      vthres=0.,     
                     ignore_norm=False,
                     force_normalize=True,)
    
        '''
        ['No benefits', 'Responder', 'TIDE', 'IFNG', 'MSI Score', 'CD274', 'CD8',
        'CTL.flag', 'Dysfunction', 'Exclusion', 'MDSC', 'CAF', 'TAM M2', 'CTL']
        '''
        return result['TIDE'].to_frame(name=self.name)



