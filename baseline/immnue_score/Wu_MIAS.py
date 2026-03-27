import numpy as np
import pandas as pd
from .base import GeneSetScorer
from .scorer import avgAbundance, pcaAbundance, origAbundance, ssGSEA
from .markers import MARKERS


class Wu_MIAS(GeneSetScorer):

    def __init__(self, cancer_type = 'SKCM', drug_target = 'PD1'):
        self.cancer_type = cancer_type
        self.drug_target = drug_target
        gs = MARKERS.loc['Wu_MIAS']
        self.name = gs.name
        self.reference = gs.Reference
        self.description = gs.Description
        self.gene_set = gs.Genes.split(':')
        self.gs = gs
        
    def _make_scorer(self):
        return ssGSEA(self.gene_set, self.name)

    def __call__(self, df_tpm):
        ssgsea = ssGSEA(self.gene_set, self.name)
        return ssgsea.fit_transform(df_tpm)

