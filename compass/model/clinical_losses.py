"""
Clinical Feature Integration Loss Functions for BioCOMPASS

This module implements loss functions for integrating clinical biomarkers
into the COMPASS model for improved immunotherapy response prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ConceptAlignmentLoss(nn.Module):
    """Aligns COMPASS concepts with external biomarkers.

    Three alignment modes:
    1. 'manual': Uses semantic mapping between concepts and biomarkers
    2. 'correlation': Auto-discovers alignments via correlation
    3. 'learnable': Uses learnable projection to align representations

    Args:
        concept_dim: Dimension of COMPASS concepts (default: 44)
        biomarker_dim: Dimension of cell type biomarkers (default: 62)
        alignment_mode: One of 'manual', 'correlation', 'learnable'
        mapping_file: Path to JSON file with semantic mappings (for 'manual' mode)
    """

    def __init__(self,
                 concept_dim: int = 44,
                 biomarker_dim: int = 62,
                 alignment_mode: str = 'manual',
                 mapping_file: Optional[str] = None):
        super().__init__()

        self.concept_dim = concept_dim
        self.biomarker_dim = biomarker_dim
        self.alignment_mode = alignment_mode

        if alignment_mode == 'manual':
            # Load semantic mapping
            if mapping_file is None:
                import os
                # Default mapping file
                cwd = os.path.dirname(os.path.dirname(__file__))  # compass directory
                mapping_file = os.path.join(cwd, 'config', 'concept_biomarker_mapping.json')

            self.mapping = self._load_semantic_mapping(mapping_file)

        elif alignment_mode == 'learnable':
            # Learnable projection from concepts to biomarkers
            from .clinical_modules import ConceptAligner
            self.concept_proj = ConceptAligner(concept_dim, biomarker_dim)

        elif alignment_mode != 'correlation':
            raise ValueError(f"Unknown alignment_mode: {alignment_mode}. Must be 'manual', 'correlation', or 'learnable'")

    def _load_semantic_mapping(self, mapping_file: str) -> Dict[int, List[int]]:
        """Load semantic mapping from JSON file.

        Returns:
            Dictionary mapping concept_idx -> list of biomarker indices
        """
        import json
        with open(mapping_file, 'r') as f:
            data = json.load(f)

        # Convert to concept_idx -> biomarker_indices mapping
        # We'll need biomarker names to indices conversion
        # For now, store biomarker names
        mapping = {}
        for concept_idx_str, concept_data in data['mappings'].items():
            concept_idx = int(concept_idx_str)
            biomarker_names = concept_data['biomarkers']
            if biomarker_names:  # Only store if has biomarkers
                mapping[concept_idx] = biomarker_names

        return mapping

    def _manual_alignment(self, concepts: torch.Tensor, biomarkers: torch.Tensor,
                         biomarker_names: List[str]) -> torch.Tensor:
        """Compute alignment loss using manual semantic mapping.

        Args:
            concepts: [B, concept_dim] COMPASS concepts
            biomarkers: [B, biomarker_dim] cell type biomarkers
            biomarker_names: List of biomarker feature names

        Returns:
            Alignment loss (correlation-based)
        """
        # Build name to index mapping
        biomarker_name_to_idx = {name: idx for idx, name in enumerate(biomarker_names)}

        loss = 0.0
        num_pairs = 0

        for concept_idx, target_biomarker_names in self.mapping.items():
            if concept_idx >= concepts.size(1):
                continue

            # Get indices of target biomarkers
            biomarker_indices = []
            for name in target_biomarker_names:
                if name in biomarker_name_to_idx:
                    biomarker_indices.append(biomarker_name_to_idx[name])

            if not biomarker_indices:
                continue

            # Concept vector
            concept_vec = concepts[:, concept_idx]  # [B]

            # Average of mapped biomarkers
            biomarker_vec = biomarkers[:, biomarker_indices].mean(dim=1)  # [B]

            # Pearson correlation loss: 1 - |correlation|
            # We want high absolute correlation
            corr = self._pearson_correlation(concept_vec, biomarker_vec)
            loss += 1.0 - torch.abs(corr)
            num_pairs += 1

        if num_pairs > 0:
            loss = loss / num_pairs

        return loss

    def _correlation_alignment(self, concepts: torch.Tensor, biomarkers: torch.Tensor) -> torch.Tensor:
        """Compute alignment loss by maximizing overall correlation.

        Args:
            concepts: [B, concept_dim]
            biomarkers: [B, biomarker_dim]

        Returns:
            Negative mean absolute correlation
        """
        # Compute correlation matrix between concepts and biomarkers
        corr_matrix = self._correlation_matrix(concepts, biomarkers)  # [concept_dim, biomarker_dim]

        # Take max correlation for each concept
        max_corr_per_concept = torch.max(torch.abs(corr_matrix), dim=1)[0]  # [concept_dim]

        # Loss: negative mean of max correlations (want to maximize)
        loss = -max_corr_per_concept.mean()

        return loss

    def _learnable_alignment(self, concepts: torch.Tensor, biomarkers: torch.Tensor) -> torch.Tensor:
        """Compute alignment loss using learnable projection.

        Args:
            concepts: [B, concept_dim]
            biomarkers: [B, biomarker_dim]

        Returns:
            MSE loss between projected biomarkers and concepts
        """
        # Project concepts to biomarker space
        projected_biomarkers = self.concept_proj(concepts)  # [B, biomarker_dim]

        # MSE loss
        loss = F.mse_loss(projected_biomarkers, biomarkers)

        return loss

    @staticmethod
    def _pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation between two vectors.

        Args:
            x: [B] vector
            y: [B] vector

        Returns:
            Correlation coefficient (scalar)
        """
        x_centered = x - x.mean()
        y_centered = y - y.mean()

        numerator = (x_centered * y_centered).sum()
        denominator = torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())

        return numerator / (denominator + 1e-8)

    @staticmethod
    def _correlation_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute correlation matrix between two sets of features.

        Args:
            x: [B, D1] features
            y: [B, D2] features

        Returns:
            corr_matrix: [D1, D2] pairwise correlations
        """
        # Center features
        x_centered = x - x.mean(dim=0, keepdim=True)  # [B, D1]
        y_centered = y - y.mean(dim=0, keepdim=True)  # [B, D2]

        # Covariance matrix
        cov_matrix = torch.mm(x_centered.t(), y_centered) / (x.size(0) - 1)  # [D1, D2]

        # Standard deviations
        x_std = torch.sqrt((x_centered ** 2).sum(dim=0) / (x.size(0) - 1))  # [D1]
        y_std = torch.sqrt((y_centered ** 2).sum(dim=0) / (y.size(0) - 1))  # [D2]

        # Correlation matrix
        corr_matrix = cov_matrix / (x_std.unsqueeze(1) * y_std.unsqueeze(0) + 1e-8)

        return corr_matrix

    def forward(self,
                concepts: torch.Tensor,
                biomarkers: torch.Tensor,
                biomarker_names: Optional[List[str]] = None) -> torch.Tensor:
        """Compute concept alignment loss.

        Args:
            concepts: [B, concept_dim] COMPASS concepts
            biomarkers: [B, biomarker_dim] cell type biomarkers
            biomarker_names: List of biomarker names (required for 'manual' mode)

        Returns:
            Alignment loss
        """
        if self.alignment_mode == 'manual':
            if biomarker_names is None:
                raise ValueError("biomarker_names required for manual alignment mode")
            return self._manual_alignment(concepts, biomarkers, biomarker_names)
        elif self.alignment_mode == 'correlation':
            return self._correlation_alignment(concepts, biomarkers)
        elif self.alignment_mode == 'learnable':
            return self._learnable_alignment(concepts, biomarkers)
        else:
            raise ValueError(f"Unknown alignment_mode: {self.alignment_mode}")


class PathwayConsistencyLoss(nn.Module):
    """Predicts pathway scores from gene encodings for auxiliary regularization.

    Regularizes the encoder to capture pathway-relevant information by
    predicting external pathway activity scores (CTLA4, PD1 pathways).

    Args:
        encoding_dim: Dimension of gene encodings (default: 32)
        num_pathways: Number of pathway scores (default: 42)
        hidden_dim: Hidden dimension for predictor head (default: 64)
    """

    def __init__(self, encoding_dim: int = 32, num_pathways: int = 42, hidden_dim: int = 64):
        super().__init__()

        self.encoding_dim = encoding_dim
        self.num_pathways = num_pathways

        # Import here to avoid circular dependency
        from .clinical_modules import PathwayPredictorHead
        self.pathway_head = PathwayPredictorHead(encoding_dim, num_pathways, hidden_dim)

    def forward(self, gene_encoding: torch.Tensor, pathway_targets: torch.Tensor) -> torch.Tensor:
        """Compute pathway consistency loss.

        Args:
            gene_encoding: [B, L, encoding_dim] gene token encodings from transformer
            pathway_targets: [B, num_pathways] target pathway scores

        Returns:
            MSE loss between predicted and target pathway scores
        """
        # Predict pathway scores
        pathway_pred = self.pathway_head(gene_encoding)  # [B, num_pathways]

        # MSE loss
        loss = F.mse_loss(pathway_pred, pathway_targets)

        return loss


class AuxiliaryTaskLoss(nn.Module):
    """Multi-task auxiliary prediction for TIDE/IPRES/phenotype scores.

    Predicts established biomarker scores alongside main task for regularization
    and improved generalization.

    Args:
        concept_dim: Dimension of COMPASS concepts (default: 44)
        aux_task_dims: Dictionary mapping task name to number of targets
        hidden_dim: Hidden dimension for auxiliary heads (default: 32)
    """

    def __init__(self,
                 concept_dim: int = 44,
                 aux_task_dims: Optional[Dict[str, int]] = None,
                 hidden_dim: int = 32):
        super().__init__()

        self.concept_dim = concept_dim

        if aux_task_dims is None:
            # Default: TIDE (10), IPRES (4), phenotype (16)
            aux_task_dims = {
                'TIDE': 10,
                'IPRES': 4,
                'phenotype': 16
            }

        self.aux_task_dims = aux_task_dims

        # Import here to avoid circular dependency
        from .clinical_modules import AuxiliaryDecoderHead

        # Create auxiliary heads
        self.aux_heads = nn.ModuleDict()
        for task_name, num_targets in aux_task_dims.items():
            self.aux_heads[task_name] = AuxiliaryDecoderHead(
                concept_dim=concept_dim,
                num_targets=num_targets,
                hidden_dim=hidden_dim
            )

    def forward(self,
                concepts: torch.Tensor,
                targets_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute auxiliary task losses.

        Args:
            concepts: [B, concept_dim] COMPASS concept representations
            targets_dict: Dictionary mapping task name to [B, num_targets] targets

        Returns:
            Tuple of:
                - total_loss: Sum of all auxiliary losses
                - predictions_dict: Dictionary of auxiliary predictions
        """
        total_loss = 0.0
        predictions_dict = {}

        for task_name, aux_head in self.aux_heads.items():
            if task_name not in targets_dict:
                continue

            # Predict
            pred = aux_head(concepts)  # [B, num_targets]
            predictions_dict[task_name] = pred

            # Compute loss
            target = targets_dict[task_name]
            loss = F.mse_loss(pred, target)
            total_loss = total_loss + loss

        return total_loss, predictions_dict
