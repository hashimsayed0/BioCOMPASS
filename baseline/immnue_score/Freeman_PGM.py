

import numpy as np
import pandas as pd
from .base import GeneSetScorer
from .scorer import avgAbundance, pcaAbundance, origAbundance, ssGSEA
from .markers import MARKERS


class Freeman_PGM(GeneSetScorer):
    '''
    Paper: Freeman S S, Sade-Feldman M, Kim J, et al. Combined tumor and immune signals from genomes or transcriptomes predict outcomes of checkpoint inhibition in melanoma[J]. Cell Reports Medicine, 2022, 3(2).
    
    Paired gene markers (PGM) discovered by Freeman from R/Long-OS and NR/Short-OS accosciations.
    '''
    def __init__(self, cancer_type = 'SKCM', drug_target = 'PD1'):

        self.cancer_type = cancer_type
        self.drug_target = drug_target
        gs = MARKERS.loc['Freeman_PGM']
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


