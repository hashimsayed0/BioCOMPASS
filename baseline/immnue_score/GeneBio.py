import numpy as np
import pandas as pd
from .base import GeneSetScorer
from .scorer import avgAbundance, pcaAbundance, origAbundance, ssGSEA
from .markers import MARKERS


class GeneBio(GeneSetScorer):
    '''
    Immunotherapy target marker: PD1_PD-L1_CTLA4, Kong J H, et al. Nature communications, 2022, 13(1): 3703.
    '''

    def __init__(self, cancer_type = 'SKCM', drug_target = 'PD1'):

        self.cancer_type = cancer_type
        self.drug_target = drug_target
        gs = MARKERS.loc['CD8']
        self.name = gs.name
        self.reference = gs.Reference
        self.description = gs.Description
        self.gene_set = gs.Genes.split(':')
        self.gs = gs

        
    def _make_scorer(self):
        return origAbundance(self.gene_set, self.name)

    def __call__(self, df_tpm):
        org = origAbundance(self.gene_set, self.name)
        return org.fit_transform(df_tpm)
