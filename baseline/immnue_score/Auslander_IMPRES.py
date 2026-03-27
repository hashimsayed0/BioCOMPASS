import numpy as np
import pandas as pd
from .base import GeneSetScorer
from .scorer import avgAbundance, pcaAbundance, origAbundance, ssGSEA
from .markers import MARKERS


class Auslander_IMPRES(GeneSetScorer):
    
    def __init__(self, cancer_type = 'SKCM', drug_target = 'PD1'):
        """
        Calculates the IMPRES score based on predifined 15 gene-gene pairs.
        """
        self.cancer_type = cancer_type
        self.drug_target = drug_target
        
        dfgs = MARKERS[MARKERS.index.map(lambda x:'IMPRES' in x)]
        gs1 = dfgs.iloc[0]
        gs2 = dfgs.iloc[1]

        self.name = 'Auslander_IMPRES'
        self.reference = gs1.Reference
        self.description = gs1.Description
        
        gs1 = gs1.Genes.split(':')
        gs2 = gs2.Genes.split(':')

        self.pair_set = [i+'_'+ j for i, j in zip(gs1, gs2)]

        self.gene_set = gs1 + gs2
        
        self.gs = dfgs
     


    def fit(self, df_train, seed=42):
        return self

    def transform(self, df_tpm):
        return self(df_tpm)

    def __call__(self, df_tpm):

        log2tpm_df = np.log2(df_tpm + 1)
        pidx = log2tpm_df.index #pateint index
        fidx = log2tpm_df.columns #
        
        pairwise_interactions = pd.DataFrame(np.empty((log2tpm_df.shape[0],len(self.pair_set)),
                                                      dtype=object),
                                             columns=self.pair_set,
                                             index= pidx)
        for feature in self.pair_set: #for each provided feature
            gene1,gene2=feature.split('_') #split provided gene1_gene2, where feature is gene1>gene2 by convention
            pairwise_interactions[feature] = np.array(1*(log2tpm_df[gene1].astype(float) > log2tpm_df[gene2].astype(float))) 
            #for each sample, calculate if feature is present
        
        score = pairwise_interactions.sum(axis=1) #sum present features for each sample
        score.name = self.name
        
        return score.to_frame()
