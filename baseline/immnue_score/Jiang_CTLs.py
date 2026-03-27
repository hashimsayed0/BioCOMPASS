import numpy as np
import pandas as pd
from .base import GeneSetScorer
from .scorer import avgAbundance, pcaAbundance, origAbundance, ssGSEA
from .markers import MARKERS



class Jiang_CTLs(GeneSetScorer):
    '''
    CTL: cytotoxic T lymphocytes markers
    Rooney MS, Shukla SA, Wu CJ, Getz G & Hacohen N Molecular and genetic properties of tumors associated with local immune cytolytic activity. Cell 160, 48–61 (2015). 
    Jiang P, Gu S, Pan D, et al. Signatures of T cell dysfunction and exclusion predict cancer immunotherapy response[J]. Nature medicine, 2018, 24(10): 1550-1558.
    Kong J H, Ha D, Lee J, et al. Network-based machine learning approach to predict immunotherapy response in cancer patients[J]. Nature communications, 2022, 13(1): 3703.
    
    '''

    def __init__(self, cancer_type = 'SKCM', drug_target = 'PD1'):

        self.cancer_type = cancer_type
        self.drug_target = drug_target
        gs = MARKERS.loc['Jiang_CTLs']
        self.name = gs.name
        self.reference = gs.Reference
        self.description = gs.Description
        self.gene_set = gs.Genes.split(':')
        self.gs = gs

        
    def _make_scorer(self):
        return avgAbundance(self.gene_set, self.name)

    def __call__(self, df_tpm):
        avg = avgAbundance(self.gene_set, self.name)
        return avg.fit_transform(df_tpm)
