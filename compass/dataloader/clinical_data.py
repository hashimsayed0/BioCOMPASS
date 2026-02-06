"""
Clinical feature loading and preprocessing for COMPASS.

This module handles loading and preprocessing of clinical features
from the iAtlas dataset, including:
- Treatment indicators (binary)
- Cell type biomarkers (continuous)
- Pathway activity scores (continuous)
- Predictive biomarkers (TIDE, IPRES, etc.)
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


class ClinicalFeatureLoader:
    """Loads and preprocesses clinical features from TSV file.

    Features are categorized into:
    - Treatment indicators (4 binary features)
    - Cell type biomarkers (62 continuous features)
    - Pathway scores (42 continuous features)
    - Auxiliary biomarkers (31 continuous features - TIDE, IPRES, phenotype)
    """

    # Define feature categories
    TREATMENT_FEATURES = ['aPD1_Tx', 'aCTLA4_Tx', 'aCTLA4_aPD1_Tx', 'Prior_aCTLA4_Tx']

    CELL_TYPE_BIOMARKERS = [
        # B cell biomarkers
        'IglesiaVincent_BCell', 'Palmer_BCell', 'Bindea_BCells', 'Schmidt_BCell',
        'GO_BCR_Signaling', 'Fan_IGG', 'Rody_TNBC_BCell', 'Vincent_Plasma_Cells',
        # T cell biomarkers
        'Palmer_CD8', 'IglesiaVincent_CD8', 'Bindea_CD8_TCells', 'IglesiaVincent_TCell',
        'Palmer_TCell', 'Bindea_TCells', 'GO_TCR_Signaling', 'Rody_TNBC_TCell',
        'Bindea_THelper', 'Bindea_Th1_Cells', 'Bindea_Th2_Cells', 'Bindea_Th17_Cells',
        'Bindea_TReg', 'Bindea_Cytotoxic_Cells', 'Bindea_Tcm', 'Bindea_Tem',
        'Bindea_TFH', 'Bindea_Tgd', 'Rody_LCK', 'IglesiaVincent_MacTh1', 'TIDE_CD8',
        # Myeloid biomarkers
        'IglesiaVincent_CD68', 'Beck_Mac_CSF1', 'Bindea_Macrophages', 'Bindea_aDC',
        'Bindea_DC', 'Bindea_iDC', 'Bindea_Neutrophils', 'Bindea_pDC', 'CSF1_Response',
        # Innate immune biomarkers
        'Bindea_Eosinophils', 'Bindea_Mast_Cells', 'Bindea_NK_CD56bright',
        'Bindea_NK_CD56dim', 'Bindea_NK_Cells'
    ]

    PATHWAY_BIOMARKERS = [
        # CTLA4 pathway scores (28 features)
        'BIOCARTA_CTLA4_V_Palmer_CD8', 'BIOCARTA_CTLA4_V_IglesiaVincent_CD8',
        'BIOCARTA_CTLA4_V_Bindea_CD8_TCells', 'BIOCARTA_CTLA4_V_IglesiaVincent_TCell',
        'BIOCARTA_CTLA4_V_Palmer_TCell', 'BIOCARTA_CTLA4_V_Bindea_TCells',
        'BIOCARTA_CTLA4_V_GO_TCR_Signaling', 'BIOCARTA_CTLA4_V_Rody_TNBC_TCell',
        'BIOCARTA_CTLA4_V_Bindea_THelper', 'BIOCARTA_CTLA4_V_Bindea_Th1_Cells',
        'BIOCARTA_CTLA4_V_Bindea_Th2_Cells', 'BIOCARTA_CTLA4_V_Bindea_Th17_Cells',
        'BIOCARTA_CTLA4_V_Bindea_TReg', 'BIOCARTA_CTLA4_V_Bindea_Cytotoxic_Cells',
        'REACTOME_CTLA4_V_Palmer_CD8', 'REACTOME_CTLA4_V_IglesiaVincent_CD8',
        'REACTOME_CTLA4_V_Bindea_CD8_TCells', 'REACTOME_CTLA4_V_IglesiaVincent_TCell',
        'REACTOME_CTLA4_V_Palmer_TCell', 'REACTOME_CTLA4_V_Bindea_TCells',
        'REACTOME_CTLA4_V_GO_TCR_Signaling', 'REACTOME_CTLA4_V_Rody_TNBC_TCell',
        'REACTOME_CTLA4_V_Bindea_THelper', 'REACTOME_CTLA4_V_Bindea_Th1_Cells',
        'REACTOME_CTLA4_V_Bindea_Th2_Cells', 'REACTOME_CTLA4_V_Bindea_Th17_Cells',
        'REACTOME_CTLA4_V_Bindea_TReg', 'REACTOME_CTLA4_V_Bindea_Cytotoxic_Cells',
        # PD1 pathway scores (14 features)
        'REACTOME_PD1_V_Palmer_CD8', 'REACTOME_PD1_V_IglesiaVincent_CD8',
        'REACTOME_PD1_V_Bindea_CD8_TCells', 'REACTOME_PD1_V_IglesiaVincent_TCell',
        'REACTOME_PD1_V_Palmer_TCell', 'REACTOME_PD1_V_Bindea_TCells',
        'REACTOME_PD1_V_GO_TCR_Signaling', 'REACTOME_PD1_V_Rody_TNBC_TCell',
        'REACTOME_PD1_V_Bindea_THelper', 'REACTOME_PD1_V_Bindea_Th1_Cells',
        'REACTOME_PD1_V_Bindea_Th2_Cells', 'REACTOME_PD1_V_Bindea_Th17_Cells',
        'REACTOME_PD1_V_Bindea_TReg', 'REACTOME_PD1_V_Bindea_Cytotoxic_Cells'
    ]

    AUXILIARY_BIOMARKERS = {
        'TIDE': ['TIDE', 'TIDE_IFNG', 'TIDE_MSI', 'TIDE_CD274', 'TIDE_Dysfunction',
                 'TIDE_Exclusion', 'TIDE_MDSC', 'TIDE_CAF', 'TIDE_TAM_M2', 'TIDE_CTL'],
        'IPRES': ['Hugo_IPRES26', 'Hugo_IPRES22', 'Hugo_IPRES08', 'Hugo_IPRES06'],
        'phenotype': ['Rody_IL8', 'ICR', 'IE_Specific', 'ID_Specific', 'Miracle',
                     'KardosChai_ImSuppress', 'Prat_Claudin', 'KardosChai_EMT_DOWN',
                     'KardosChai_EMT_UP', 'Chan_TIC', 'LIexpression_Score', 'TGFB_Score',
                     'Module3_IFN_Score', 'Chang_Serum_Response_Up', 'Cytolytic_Score', 'IMPRES']
    }

    def __init__(self, clinical_features_file: str, normalize: bool = True):
        """Initialize clinical feature loader.

        Args:
            clinical_features_file: Path to TSV file with clinical features
            normalize: Whether to normalize continuous features (z-score)
        """
        self.clinical_features_file = clinical_features_file
        self.normalize = normalize

        # Load data
        self.df = pd.read_csv(clinical_features_file, sep='\t', index_col=0)

        # Compute normalization statistics on training data (set later)
        self.treatment_mean = None
        self.treatment_std = None
        self.cell_type_mean = None
        self.cell_type_std = None
        self.pathway_mean = None
        self.pathway_std = None
        self.auxiliary_mean = {}
        self.auxiliary_std = {}

    def fit(self, sample_ids: List[str]):
        """Compute normalization statistics on training samples.

        Args:
            sample_ids: List of training sample IDs
        """
        if not self.normalize:
            return

        train_samples = [s for s in sample_ids if s in self.df.index]
        if len(train_samples) == 0:
            raise ValueError("No training samples found in clinical features file")

        df_train = self.df.loc[train_samples]

        # Treatment features (binary, but store stats for consistency)
        if all(f in df_train.columns for f in self.TREATMENT_FEATURES):
            treatment_df = df_train[self.TREATMENT_FEATURES].fillna(0)
            # Convert boolean to float
            treatment_df = treatment_df.astype(float)
            self.treatment_mean = treatment_df.mean().values
            self.treatment_std = treatment_df.std().values + 1e-6

        # Cell type biomarkers
        cell_cols = [c for c in self.CELL_TYPE_BIOMARKERS if c in df_train.columns]
        if cell_cols:
            cell_df = df_train[cell_cols].fillna(0)
            self.cell_type_mean = cell_df.mean().values
            self.cell_type_std = cell_df.std().values + 1e-6

        # Pathway biomarkers
        pathway_cols = [c for c in self.PATHWAY_BIOMARKERS if c in df_train.columns]
        if pathway_cols:
            pathway_df = df_train[pathway_cols].fillna(0)
            self.pathway_mean = pathway_df.mean().values
            self.pathway_std = pathway_df.std().values + 1e-6

        # Auxiliary biomarkers
        for aux_name, aux_cols in self.AUXILIARY_BIOMARKERS.items():
            aux_cols_present = [c for c in aux_cols if c in df_train.columns]
            if aux_cols_present:
                aux_df = df_train[aux_cols_present].fillna(0)
                self.auxiliary_mean[aux_name] = aux_df.mean().values
                self.auxiliary_std[aux_name] = aux_df.std().values + 1e-6

    def transform(self, sample_ids: List[str]) -> Dict[str, np.ndarray]:
        """Extract and normalize clinical features for given samples.

        Args:
            sample_ids: List of sample IDs to extract features for

        Returns:
            Dictionary with keys:
                - 'treatment': [N, 4] binary treatment indicators
                - 'cell_type': [N, 62] cell type biomarker scores
                - 'pathway': [N, 42] pathway activity scores
                - 'auxiliary_TIDE': [N, 10] TIDE scores
                - 'auxiliary_IPRES': [N, 4] IPRES scores
                - 'auxiliary_phenotype': [N, 16] phenotype scores
        """
        valid_samples = [s for s in sample_ids if s in self.df.index]
        if len(valid_samples) == 0:
            raise ValueError("No samples found in clinical features file")

        df_subset = self.df.loc[valid_samples]
        features = {}

        # Treatment features (binary)
        treatment_cols = [c for c in self.TREATMENT_FEATURES if c in df_subset.columns]
        if treatment_cols:
            treatment_df = df_subset[treatment_cols].fillna(False)
            # Convert to float (True -> 1.0, False -> 0.0)
            treatment = treatment_df.astype(float).values
            features['treatment'] = treatment
        else:
            features['treatment'] = np.zeros((len(valid_samples), 4))

        # Cell type biomarkers
        cell_cols = [c for c in self.CELL_TYPE_BIOMARKERS if c in df_subset.columns]
        if cell_cols:
            cell_df = df_subset[cell_cols].fillna(0)
            cell_type = cell_df.values
            if self.normalize and self.cell_type_mean is not None:
                cell_type = (cell_type - self.cell_type_mean) / self.cell_type_std
            features['cell_type'] = cell_type
        else:
            features['cell_type'] = np.zeros((len(valid_samples), len(self.CELL_TYPE_BIOMARKERS)))

        # Pathway biomarkers
        pathway_cols = [c for c in self.PATHWAY_BIOMARKERS if c in df_subset.columns]
        if pathway_cols:
            pathway_df = df_subset[pathway_cols].fillna(0)
            pathway = pathway_df.values
            if self.normalize and self.pathway_mean is not None:
                pathway = (pathway - self.pathway_mean) / self.pathway_std
            features['pathway'] = pathway
        else:
            features['pathway'] = np.zeros((len(valid_samples), len(self.PATHWAY_BIOMARKERS)))

        # Auxiliary biomarkers
        for aux_name, aux_cols in self.AUXILIARY_BIOMARKERS.items():
            aux_cols_present = [c for c in aux_cols if c in df_subset.columns]
            if aux_cols_present:
                aux_df = df_subset[aux_cols_present].fillna(0)
                aux = aux_df.values
                if self.normalize and aux_name in self.auxiliary_mean:
                    aux = (aux - self.auxiliary_mean[aux_name]) / self.auxiliary_std[aux_name]
                features[f'auxiliary_{aux_name}'] = aux
            else:
                features[f'auxiliary_{aux_name}'] = np.zeros((len(valid_samples), len(aux_cols)))

        return features

    def fit_transform(self, sample_ids: List[str]) -> Dict[str, np.ndarray]:
        """Fit normalization and transform in one call.

        Args:
            sample_ids: Training sample IDs

        Returns:
            Dictionary of clinical features
        """
        self.fit(sample_ids)
        return self.transform(sample_ids)

    def get_sample_ids(self) -> List[str]:
        """Get all available sample IDs."""
        return self.df.index.tolist()

    def to_tensor_dict(self, features_dict: Dict[str, np.ndarray], device='cpu') -> Dict[str, torch.Tensor]:
        """Convert numpy arrays to torch tensors.

        Args:
            features_dict: Dictionary of numpy arrays
            device: Device to place tensors on

        Returns:
            Dictionary of torch tensors
        """
        tensor_dict = {}
        for key, arr in features_dict.items():
            tensor_dict[key] = torch.from_numpy(arr).float().to(device)
        return tensor_dict


def load_clinical_features(clinical_features_file: str,
                           train_samples: List[str],
                           test_samples: Optional[List[str]] = None,
                           normalize: bool = True) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """Convenience function to load and normalize clinical features.

    Args:
        clinical_features_file: Path to clinical features TSV
        train_samples: Training sample IDs (used for normalization)
        test_samples: Optional test sample IDs
        normalize: Whether to normalize continuous features

    Returns:
        Tuple of (train_features, test_features) dictionaries
    """
    loader = ClinicalFeatureLoader(clinical_features_file, normalize=normalize)

    # Fit on training data
    train_features = loader.fit_transform(train_samples)

    # Transform test data if provided
    test_features = None
    if test_samples is not None:
        test_features = loader.transform(test_samples)

    return train_features, test_features