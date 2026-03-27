import numpy as np
import pandas as pd
from .base import GeneSetScorer
from .scorer import avgAbundance, pcaAbundance, origAbundance, ssGSEA
from .markers import MARKERS



class PDL1(GeneSetScorer):
    '''
    Single target marker PDL1
    '''

    def __init__(self, cancer_type = 'SKCM', drug_target = 'PD1'):

        self.cancer_type = cancer_type
        self.drug_target = drug_target
        gs = MARKERS.loc['PDL1']
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
