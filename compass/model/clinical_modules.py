"""
Clinical feature integration modules for COMPASS.

This module implements various strategies for integrating clinical features:
1. TreatmentGating: Modulates concepts based on treatment type
2. BiomarkerGuidedAttention: Cross-attention between biomarkers and gene encodings
3. BiomarkerEncoder: Encodes continuous biomarker features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class TreatmentGating(nn.Module):
    """Modulates concept representations based on treatment type.

    Treatment indicators (PD1, CTLA4, combo) are embedded and used to compute
    gating weights that modulate the 44 biological concepts.

    Args:
        num_treatments: Number of treatment types (default: 4)
        concept_dim: Dimension of concept representations (default: 44)
        hidden_dim: Hidden dimension for gating network (default: 32)
    """

    def __init__(self, num_treatments: int = 4, concept_dim: int = 44, hidden_dim: int = 32):
        super(TreatmentGating, self).__init__()

        self.num_treatments = num_treatments
        self.concept_dim = concept_dim
        self.hidden_dim = hidden_dim

        # Embedding layer for treatment combinations
        # We use 2^4 = 16 possible combinations of 4 binary treatment indicators
        self.treatment_embed = nn.Embedding(16, hidden_dim)

        # Gating network: maps treatment embedding to concept-wise gates
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, concept_dim),
            nn.Sigmoid()  # Gates in [0, 1]
        )

    def forward(self, concepts: torch.Tensor, treatment_indicators: torch.Tensor) -> torch.Tensor:
        """Apply treatment-aware gating to concepts.

        Args:
            concepts: [B, concept_dim] biological concept representations
            treatment_indicators: [B, 4] binary indicators for treatments

        Returns:
            gated_concepts: [B, concept_dim] modulated concepts
        """
        # Convert binary indicators to integer index (0-15)
        # Each combination of 4 binary flags maps to unique index
        treatment_idx = (treatment_indicators * torch.tensor([8, 4, 2, 1], device=treatment_indicators.device)).sum(dim=1).long()

        # Embed treatment combination
        treat_embed = self.treatment_embed(treatment_idx)  # [B, hidden_dim]

        # Compute gating weights
        gate = self.gate_net(treat_embed)  # [B, concept_dim]

        # Apply element-wise gating
        gated_concepts = concepts * gate

        return gated_concepts


class BiomarkerEncoder(nn.Module):
    """Encodes continuous biomarker features into a latent representation.

    Args:
        biomarker_dim: Input dimension of biomarker features
        hidden_dim: Hidden dimension
        output_dim: Output embedding dimension
    """

    def __init__(self, biomarker_dim: int, hidden_dim: int = 64, output_dim: int = 32):
        super(BiomarkerEncoder, self).__init__()

        self.biomarker_dim = biomarker_dim
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(biomarker_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, biomarkers: torch.Tensor) -> torch.Tensor:
        """Encode biomarkers to latent representation.

        Args:
            biomarkers: [B, biomarker_dim] continuous biomarker features

        Returns:
            encoded: [B, output_dim] encoded biomarker representation
        """
        return self.encoder(biomarkers)


class BiomarkerGuidedAttention(nn.Module):
    """Cross-attention mechanism where biomarkers guide attention over gene encodings.

    Uses multi-head attention with biomarker embeddings as queries and
    gene encodings as keys/values.

    Args:
        gene_encoding_dim: Dimension of gene encodings (default: 32)
        biomarker_dim: Combined dimension of all biomarkers
        attention_dim: Dimension for attention mechanism (default: 32)
        num_heads: Number of attention heads (default: 4)
    """

    def __init__(self,
                 gene_encoding_dim: int = 32,
                 biomarker_dim: int = 115,
                 attention_dim: int = 32,
                 num_heads: int = 4):
        super(BiomarkerGuidedAttention, self).__init__()

        self.gene_encoding_dim = gene_encoding_dim
        self.biomarker_dim = biomarker_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads

        # Biomarker encoder: projects biomarkers to attention query space
        self.biomarker_encoder = BiomarkerEncoder(
            biomarker_dim=biomarker_dim,
            hidden_dim=64,
            output_dim=attention_dim
        )

        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Project gene encodings to attention space
        self.gene_proj = nn.Linear(gene_encoding_dim, attention_dim)

        # Output projection
        self.output_proj = nn.Linear(attention_dim, attention_dim)

    def forward(self,
                gene_encoding: torch.Tensor,
                biomarkers: torch.Tensor) -> torch.Tensor:
        """Apply biomarker-guided attention to gene encodings.

        Args:
            gene_encoding: [B, L, gene_encoding_dim] gene token encodings from transformer
            biomarkers: [B, biomarker_dim] concatenated biomarker features

        Returns:
            attended_features: [B, attention_dim] biomarker-guided gene features
        """
        batch_size = gene_encoding.size(0)

        # Encode biomarkers to query
        biomarker_query = self.biomarker_encoder(biomarkers)  # [B, attention_dim]
        biomarker_query = biomarker_query.unsqueeze(1)  # [B, 1, attention_dim]

        # Project gene encodings to key/value space
        gene_kv = self.gene_proj(gene_encoding)  # [B, L, attention_dim]

        # Cross-attention: biomarkers attend to genes
        attended, attention_weights = self.cross_attention(
            query=biomarker_query,  # [B, 1, attention_dim]
            key=gene_kv,  # [B, L, attention_dim]
            value=gene_kv  # [B, L, attention_dim]
        )

        # Output projection
        attended = attended.squeeze(1)  # [B, attention_dim]
        output = self.output_proj(attended)  # [B, attention_dim]

        return output


class PathwayPredictorHead(nn.Module):
    """Auxiliary head for predicting pathway activity scores from gene encodings.

    Used in Strategy 3 (Pathway Consistency Loss) to regularize the encoder
    to learn pathway-relevant representations.

    Args:
        encoding_dim: Dimension of gene encodings (default: 32)
        num_pathways: Number of pathway scores to predict (default: 42)
        hidden_dim: Hidden dimension (default: 64)
    """

    def __init__(self, encoding_dim: int = 32, num_pathways: int = 42, hidden_dim: int = 64):
        super(PathwayPredictorHead, self).__init__()

        self.encoding_dim = encoding_dim
        self.num_pathways = num_pathways

        self.predictor = nn.Sequential(
            nn.Linear(encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_pathways)
        )

    def forward(self, gene_encoding: torch.Tensor) -> torch.Tensor:
        """Predict pathway scores from pooled gene encoding.

        Args:
            gene_encoding: [B, L, encoding_dim] gene encodings

        Returns:
            pathway_scores: [B, num_pathways] predicted pathway scores
        """
        # Pool gene encodings (mean over sequence length)
        pooled = gene_encoding.mean(dim=1)  # [B, encoding_dim]

        # Predict pathway scores
        pathway_scores = self.predictor(pooled)  # [B, num_pathways]

        return pathway_scores


class AuxiliaryDecoderHead(nn.Module):
    """Auxiliary decoder head for predicting biomarker scores from concepts.

    Used in Strategy 4 (Multi-Task Auxiliary Prediction) for predicting
    TIDE, IPRES, and phenotype scores alongside the main task.

    Args:
        concept_dim: Dimension of concept representations (default: 44)
        num_targets: Number of targets to predict
        hidden_dim: Hidden dimension (default: 32)
    """

    def __init__(self, concept_dim: int = 44, num_targets: int = 1, hidden_dim: int = 32):
        super(AuxiliaryDecoderHead, self).__init__()

        self.concept_dim = concept_dim
        self.num_targets = num_targets

        self.decoder = nn.Sequential(
            nn.Linear(concept_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_targets)
        )

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        """Predict auxiliary targets from concepts.

        Args:
            concepts: [B, concept_dim] biological concept representations

        Returns:
            predictions: [B, num_targets] auxiliary predictions
        """
        return self.decoder(concepts)


class ConceptAligner(nn.Module):
    """Learnable projection for aligning concepts with biomarkers.

    Used in Strategy 2 when alignment_mode='learnable'. Projects COMPASS concepts
    to biomarker space for alignment.

    Args:
        concept_dim: Dimension of COMPASS concepts (default: 44)
        biomarker_dim: Dimension of biomarker space (default: 62)
    """

    def __init__(self, concept_dim: int = 44, biomarker_dim: int = 62):
        super(ConceptAligner, self).__init__()

        self.concept_dim = concept_dim
        self.biomarker_dim = biomarker_dim

        # Learnable projection matrix
        self.projection = nn.Linear(concept_dim, biomarker_dim, bias=False)

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        """Project concepts to biomarker space.

        Args:
            concepts: [B, concept_dim] COMPASS concept representations

        Returns:
            projected: [B, biomarker_dim] projected representations
        """
        return self.projection(concepts)