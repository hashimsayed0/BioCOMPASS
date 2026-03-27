import numpy as np
import pandas as pd
from .base import GeneSetScorer
from .scorer import avgAbundance, pcaAbundance, origAbundance, ssGSEA
from .markers import MARKERS


class Davoli_CIS(GeneSetScorer):
    
    '''
    Davoli cytotoxic immune signatures (CIS), 
    
    Paper: 
    =============
    Davoli et al., Science 2017
    
    Description:
    =============
    Gene markers of cytotoxic immune cell infiltrates (cytotoxic CD8+ T cells and NK cells).
    
    Caclculation: 
    =============
    First, rank normalization is applied across samples. 
    Second, the average of the expression of Davoli immune signature is calculated for each sample [log2(TPM+1)].	
    
    Markers: 
    =============
    CD247, CD2, CD3E, GZMH, NKG7, PRF1, GZMK
    '''


    def __init__(self, cancer_type = 'SKCM', drug_target = 'PD1'):

        self.cancer_type = cancer_type
        self.drug_target = drug_target
        gs = MARKERS.loc['Davoli_CIS']
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
