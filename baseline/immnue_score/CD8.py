import numpy as np
import pandas as pd
from .base import GeneSetScorer
from .scorer import avgAbundance, pcaAbundance, origAbundance, ssGSEA
from .markers import MARKERS




class CD8(GeneSetScorer):
    '''
    (Average expression of CD8A + CD8B), PMID: 27301722;PMID: 30127393; PMID: 34761219
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
        return avgAbundance(self.gene_set, self.name)

    def __call__(self, df_tpm):
        avg = avgAbundance(self.gene_set, self.name)
        return avg.fit_transform(df_tpm)
