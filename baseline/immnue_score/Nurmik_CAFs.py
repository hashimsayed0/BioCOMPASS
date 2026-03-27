import numpy as np
import pandas as pd
from .base import GeneSetScorer
from .scorer import avgAbundance, pcaAbundance, origAbundance, ssGSEA
from .markers import MARKERS


'''
Cancer-associated fibroblasts (CAFs)

Nurmik M, Ullmann P, Rodriguez F, et al. In search of definitions: Cancer‐associated fibroblasts and their markers[J]. International journal of cancer, 2020, 146(4): 895-905.

Jiang P, Gu S, Pan D, et al. Signatures of T cell dysfunction and exclusion predict cancer immunotherapy response[J]. Nature medicine, 2018, 24(10): 1550-1558.

Kong J H, Ha D, Lee J, et al. Network-based machine learning approach to predict immunotherapy response in cancer patients[J]. Nature communications, 2022, 13(1): 3703.
'''


class Nurmik_CAFs(GeneSetScorer):
    '''
    Cancer-associated fibroblasts (CAFs)
    '''

    def __init__(self, cancer_type = 'SKCM', drug_target = 'PD1'):

        self.cancer_type = cancer_type
        self.drug_target = drug_target
        gs = MARKERS.loc['Nurmik_CAFs']
        self.name = gs.name
        self.reference = gs.Reference
        self.description = gs.Description
        self.gene_set = gs.Genes.split(':')
        self.gs = gs
        
    def _make_scorer(self):
        return avgAbundance(self.gene_set, self.name)

    def __call__(self, df_tpm):
        ssgsea = avgAbundance(self.gene_set, self.name)
        return ssgsea.fit_transform(df_tpm)
