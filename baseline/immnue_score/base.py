from .scorer import avgAbundance, pcaAbundance, origAbundance, ssGSEA


class GeneSetScorer:

    def __init__(self):
        self.gene_set = ['PDCD1', 'CTLA4']
        self.name = 'Base'

    # ── convenience wrappers (fit+transform on same data) ─────────────────
    def get_avg(self, df_tpm):
        score = avgAbundance(self.gene_set, self.name)
        return score.fit_transform(df_tpm)

    def get_pca(self, df_tpm):
        score = pcaAbundance(self.gene_set, self.name)
        return score.fit_transform(df_tpm)

    def get_org(self, df_tpm):
        score = origAbundance(self.gene_set, self.name)
        return score.fit_transform(df_tpm)

    def get_ssgsea(self, df_tpm):
        score = ssGSEA(self.gene_set, self.name)
        return score.fit_transform(df_tpm)

    # ── leak-free train/test interface ────────────────────────────────────
    def _make_scorer(self):
        """Return a fresh internal scorer instance. Subclasses override this."""
        raise NotImplementedError

    def fit(self, df_train, seed=42):
        """Fit the internal scorer on training data only."""
        self._seed = seed
        self._scorer = self._make_scorer()
        self._scorer.fit(df_train)
        return self

    def transform(self, df_tpm):
        """Transform df_tpm using the scorer fitted on training data."""
        return self._scorer.transform(df_tpm)
