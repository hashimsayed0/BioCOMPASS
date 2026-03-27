import numpy as np
import pandas as pd
from .base import GeneSetScorer
from .scorer import avgAbundance, pcaAbundance, origAbundance, ssGSEA
from .markers import MARKERS



class Messina_CKS(GeneSetScorer):
    '''Messina  12-chemokine signature (Principal component 1 score from PCA of expression levels of 12 chemokine signature genes)
    '''

    def __init__(self, cancer_type = 'SKCM', drug_target = 'PD1'):

        self.cancer_type = cancer_type
        self.drug_target = drug_target
        gs = MARKERS.loc['Messina_CKS']
        self.name = gs.name
        self.reference = gs.Reference
        self.description = gs.Description
        self.gene_set = gs.Genes.split(':')
        self.gs = gs

        
    def _make_scorer(self):
        return pcaAbundance(self.gene_set, self.name, random_state=getattr(self, '_seed', 42))

    def __call__(self, df_tpm):

        pcas = pcaAbundance(self.gene_set, self.name)
        return pcas.fit_transform(df_tpm)
