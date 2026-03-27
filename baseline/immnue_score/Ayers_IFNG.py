import numpy as np
import pandas as pd
import warnings
import qnorm

from .base import GeneSetScorer
from .scorer import avgAbundance, pcaAbundance, origAbundance, ssGSEA
from .markers import MARKERS

class Ayers_IFNG(GeneSetScorer):
    def __init__(self, cancer_type = 'SKCM', drug_target = 'PD1'):
        '''
        Paper:
        Ayers M, Lunceford J, Nebozhyn M, Murphy E, Loboda A, Kaufman DR, Albright A, Cheng JD, Kang SP, Shankaran V, et al: IFN-gamma-related mRNA profile predicts clinical response to PD-1 blockade. J Clin Invest 2017, 127:2930-2940.
        
        Marker:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5531419/bin/jci-127-91190-g010.jpg
        
        Calculation:
        After performance of quantile normalization, a log10 transformation was applied, and signature scores were calculated by averaging of the included genes for the IFN-γ (6-gene) and expanded immune (18-gene) signatures.
        '''
        self.cancer_type = cancer_type
        self.drug_target = drug_target
        gs = MARKERS.loc['Ayers_IFNg18']
        self.name = gs.name
        self.reference = gs.Reference
        self.description = gs.Description
        self.gene_set = gs.Genes.split(':')
        self.gs = gs
        self.ncpus = 4
        
    def fit(self, df_train, seed=42):
        return self

    def transform(self, df_tpm):
        return self(df_tpm)

    def __call__(self, df_tpm):
    
        markers_used = list(set(self.gene_set) & set(df_tpm.columns))
        markers_unused =  list(set(self.gene_set) - set(markers_used))
        warnings.warn('Markers of %s are missed and not used.' % markers_unused)
        
        ## quantile_normalize for each sample
        df_tpm_norm = qnorm.quantile_normalize(df_tpm, ncpus=self.ncpus, axis = 1)  
        df_tpm_norm_log10 = np.log10(df_tpm_norm + 1)
    
        ifng = df_tpm_norm_log10[markers_used].mean(axis=1)
        ifng.name = self.name
        ifng = ifng.to_frame()
        
        return ifng

