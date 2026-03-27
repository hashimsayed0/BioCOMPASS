import numpy as np
import pandas as pd
from .base import GeneSetScorer
from .scorer import avgAbundance, pcaAbundance, origAbundance, ssGSEA
from .markers import MARKERS

'''
Science paper for GEP weights(DOI: 10.1126/science.aar3593):
GEP_W = (CCL5=0.008346; CD27=0.072293; CD274=0.042853; CD276=-0.0239; CD8A=0.031021; CMKLR1=0.151253; CXCL9=0.074135; CXCR6=0.004313; HLA.DQA1=0.020091; HLA.DRB1=0.058806; HLA.E=0.07175; IDO1=0.060679; LAG3=0.123895; NKG7=0.075524; PDCD1LG2=0.003734; PSMB10=0.032999; STAT1=0.250229; TIGIT=0.084767)
The Nat. Comm paper (https://pubmed.ncbi.nlm.nih.gov/35013211/) calculated the GEP by ssGSEA
'''

class Cristescu_GEP(GeneSetScorer):
    def __init__(self, cancer_type = 'SKCM', drug_target = 'PD1'):
        self.cancer_type = cancer_type
        self.drug_target = drug_target
        gs = MARKERS.loc['Cristescu_GEP']
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

