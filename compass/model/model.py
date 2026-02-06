# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:31:25 2023

@author: Wanxiang Shen

"""
import torch
import torch.nn as nn
from ..encoder import TransformerEncoder, MLPEncoder
from ..decoder import ClassDecoder, RegDecoder, ProtoNetDecoder
from ..projector import DisentangledProjector, EntangledProjector


import numpy as np
import random
def fixseed(seed=42): 
    np.random.seed(seed)  
    random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.cuda.manual_seed(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




class Compass(nn.Module):

    def __init__(self,
                 input_dim,
                 task_dim,
                 task_type,
                 num_cancer_types = 33,
                 embed_dim = 32,


                 #### projections
                 proj_disentangled =  True,
                 proj_level = 'cellpathway',
                 proj_pid = False,
                 proj_cancer_type = True,
                 ref_for_task = True,
                 encoder = 'transformer',
                 encoder_dropout = 0.,
                 task_dense_layer=[24],
                 task_batch_norms = True,
                 transformer_dim = 32,
                 transformer_num_layers = 1,
                 transformer_nhead = 2,
                 transformer_pos_emb = 'learnable',
                 seed = 42,

                 #### clinical feature integration (optional)
                 use_treatment_gating = False,
                 treatment_gating_hidden_dim = 32,
                 use_biomarker_attention = False,
                 biomarker_attention_dim = 32,
                 biomarker_attention_heads = 4,
                 use_pathway_predictor = False,
                 num_pathways = 42,
                 use_auxiliary_tasks = False,
                 auxiliary_task_dims = None,  # {'TIDE': 10, 'IPRES': 4, 'phenotype': 16}

                 **encoder_kwargs
                ):
        
        '''
        input_dim:  number of tokens
        task_dim: supervised learning task dim
        task_type: {'r', 'c'}
        num_cancer_types: int, number cancer types, default 33.
        embed_dim: latent vector dim
        encoder: {'transfomer', 'flowformer', ...}
        task_dense_layer: dense layer of task
        transformer_pos_emb: {None, 'umap', 'pumap'}
        '''
        super().__init__()

        
        self.input_dim = input_dim
        self.task_dim = task_dim
        self.task_type = task_type 

        self.proj_disentangled = proj_disentangled
        self.proj_level = proj_level
        self.proj_pid = proj_pid
        self.proj_cancer_type = proj_cancer_type
        self.ref_for_task = ref_for_task
        
        self.num_cancer_types = num_cancer_types
        self.encoder = encoder
        self.encoder_dropout = encoder_dropout
        self.transformer_dim = transformer_dim
        self.transformer_num_layers = transformer_num_layers
        self.transformer_pos_emb = transformer_pos_emb
        self.transformer_nhead = transformer_nhead
        self.task_batch_norms = task_batch_norms
        self.task_dense_layer = task_dense_layer
        self.encoder_kwargs = encoder_kwargs
        self.seed = seed
        
        fixseed(seed=self.seed)
        
        self.inputencoder = TransformerEncoder(num_cancer_types=num_cancer_types,
                                               encoder_type = encoder,
                                               input_dim = input_dim, 
                                               d_model = transformer_dim, 
                                               num_layers = transformer_num_layers,
                                               nhead = transformer_nhead,
                                               dropout = encoder_dropout,
                                               pos_emb = transformer_pos_emb, 
                                               **encoder_kwargs)
        
        if proj_disentangled:
            self.latentprojector = DisentangledProjector(input_dim, transformer_dim, 
                                                         proj_pid = proj_pid, 
                                                         proj_cancer_type = proj_cancer_type)

            self.geneset_feature_name = self.latentprojector.geneset_proj_cols
            self.celltype_feature_name = self.latentprojector.cellpathway_proj_cols
            self.ref_gene_ids = self.latentprojector.ref_gene_ids
            self.ref_geneset_ids = self.latentprojector.ref_geneset_ids
            self.ref_celltype_ids = self.latentprojector.ref_celltype_ids

            if  proj_level == 'cellpathway':
                a = self.celltype_feature_name
                k = self.ref_celltype_ids
            else:
                a = self.geneset_feature_name
                k = self.ref_geneset_ids

            if not self.ref_for_task:
                self.embed_dim = len(a) - len(k)
                self.embed_feature_names = a
                self.proj_feature_names = [a[i] for i in range(len(a)) if i not in k]
            else:
                self.embed_dim = len(a)
                self.embed_feature_names = a
                self.proj_feature_names = a
                
        else:
            self.latentprojector = EntangledProjector(transformer_dim)
            self.embed_feature_names = range(len(embed_dim))
            self.embed_dim = embed_dim
            self.proj_feature_names = self.embed_feature_names

        
        model_args = {'input_dim':self.input_dim, 
                    'task_dim':self.task_dim,
                    'task_type':self.task_type, 
                      
                    'proj_level':self.proj_level,
                    'proj_pid':self.proj_pid,
                    'proj_cancer_type':self.proj_cancer_type,
                    'proj_disentangled':self.proj_disentangled,
                      
                    'embed_dim': self.embed_dim, 
                    'num_cancer_types':self.num_cancer_types,
                    'encoder':self.encoder,
                    'encoder_dropout':self.encoder_dropout,
                    'transformer_dim':self.transformer_dim,
                    'transformer_nhead':self.transformer_nhead,
                    'transformer_num_layers':self.transformer_num_layers,
                    'transformer_pos_emb':self.transformer_pos_emb,
                    'task_batch_norms':self.task_batch_norms,
                    'task_dense_layer':self.task_dense_layer,
                    'seed':self.seed,
                      
                    # Clinical feature integration
                    'use_treatment_gating':use_treatment_gating,
                    'treatment_gating_hidden_dim':treatment_gating_hidden_dim,
                    'use_biomarker_attention':use_biomarker_attention,
                    'biomarker_attention_dim':biomarker_attention_dim,
                    'biomarker_attention_heads':biomarker_attention_heads,
                    'use_pathway_predictor':use_pathway_predictor,
                    'num_pathways':num_pathways,
                    'use_auxiliary_tasks':use_auxiliary_tasks,
                    'auxiliary_task_dims':auxiliary_task_dims,
                      
               }

        model_args.update(encoder_kwargs)        
        self.model_args = model_args

        # Calculate task decoder input dimension based on biomarker attention
        task_decoder_input_dim = self.embed_dim
        if use_biomarker_attention:
            task_decoder_input_dim += biomarker_attention_dim

        ## regression task
        if task_type == 'r':
            self.taskdecoder = RegDecoder(input_dim = task_decoder_input_dim, 
                                        dense_layers = task_dense_layer, 
                                        out_dim = task_dim, 
                                        batch_norms = task_batch_norms, 
                                          seed = self.seed)
        
        ## classification task
        elif task_type == 'c':
            self.taskdecoder = ClassDecoder(input_dim = task_decoder_input_dim, 
                                          dense_layers = task_dense_layer, 
                                          out_dim = task_dim, 
                                          batch_norms = task_batch_norms,
                                            seed = self.seed)

        #for softmax classifier
        elif task_type == 'f':
            self.taskdecoder = ProtoNetDecoder(input_dim = task_decoder_input_dim,
                                                out_dim = task_dim,
                                                dense_layers = task_dense_layer,
                                                batch_norms = task_batch_norms,
                                               seed = self.seed
                                                )

        # Clinical feature integration modules (optional)
        self.use_treatment_gating = use_treatment_gating
        self.use_biomarker_attention = use_biomarker_attention
        self.use_pathway_predictor = use_pathway_predictor
        self.use_auxiliary_tasks = use_auxiliary_tasks

        if use_treatment_gating:
            from .clinical_modules import TreatmentGating
            self.treatment_gating = TreatmentGating(
                num_treatments=4,
                concept_dim=self.embed_dim,
                hidden_dim=treatment_gating_hidden_dim
            )
        else:
            self.treatment_gating = None

        if use_biomarker_attention:
            from .clinical_modules import BiomarkerGuidedAttention
            self.biomarker_attention = BiomarkerGuidedAttention(
                gene_encoding_dim=transformer_dim,
                biomarker_dim=115,  # Total biomarkers (62 cell type + 42 pathway + 11 others)
                attention_dim=biomarker_attention_dim,
                num_heads=biomarker_attention_heads
            )
        else:
            self.biomarker_attention = None

        if use_pathway_predictor:
            from .clinical_modules import PathwayPredictorHead
            self.pathway_predictor = PathwayPredictorHead(
                encoding_dim=transformer_dim,
                num_pathways=num_pathways,
                hidden_dim=64
            )
        else:
            self.pathway_predictor = None

        if use_auxiliary_tasks:
            from .clinical_modules import AuxiliaryDecoderHead
            if auxiliary_task_dims is None:
                auxiliary_task_dims = {'TIDE': 10, 'IPRES': 4, 'phenotype': 16}

            self.auxiliary_heads = nn.ModuleDict()
            for task_name, num_targets in auxiliary_task_dims.items():
                self.auxiliary_heads[task_name] = AuxiliaryDecoderHead(
                    concept_dim=self.embed_dim,
                    num_targets=num_targets,
                    hidden_dim=32
                )
        else:
            self.auxiliary_heads = None


    def forward(self, x, cohort_id=None, epoch=0,
                clinical_features=None, return_auxiliary=False):
        """Forward pass with optional clinical feature integration.

        Args:
            x: Gene expression input [B, gene_dim+1] (cancer_type + genes)
            epoch: Current training epoch (for warmup)
            clinical_features: Optional dict of clinical features:
                - 'treatment': [B, 4] treatment indicators
                - 'biomarkers': [B, 115] all biomarkers (for attention)
                - 'cell_type': [B, 62] cell type biomarkers
                - 'pathway': [B, 42] pathway scores
                - 'auxiliary_TIDE': [B, 10] TIDE scores
                - 'auxiliary_IPRES': [B, 4] IPRES scores
                - 'auxiliary_phenotype': [B, 16] phenotype scores
            return_auxiliary: Whether to return auxiliary task predictions

        Returns:
            Tuple of (embedding_tuple, main_predictions, clinical_outputs)
            If return_auxiliary=True, clinical_outputs includes 'auxiliary_predictions'
        """

        #output： B,L+2, (dataset:1, cancer:1, gene),C
        encoding = self.inputencoder(x)
        geneset_level_proj, cellpathway_level_proj = self.latentprojector(encoding)

        # task_inputs: only input the context-oriented features (for downstream task)
        # Embedding: embeddings for contrastive learning
        if self.proj_level == 'geneset':
            embedding = geneset_level_proj #B,L
            emb_ref = embedding[:, self.ref_geneset_ids]

            mask = torch.ones(embedding.shape[1], dtype=torch.bool)
            mask[self.ref_geneset_ids] = False
            emb_used = embedding[:, mask]

        elif self.proj_level == 'cellpathway':
            embedding = cellpathway_level_proj #B,L
            emb_ref = embedding[:, self.ref_celltype_ids]

            mask = torch.ones(embedding.shape[1], dtype=torch.bool)
            mask[self.ref_celltype_ids] = False
            emb_used = embedding[:, mask]
            #print(emb_used.shape)

        # Apply clinical feature integration
        enhanced_embedding = embedding
        biomarker_guided_features = None

        if clinical_features is not None:
            # Strategy 1: Treatment Gating
            if self.treatment_gating is not None and 'treatment' in clinical_features:
                enhanced_embedding = self.treatment_gating(
                    enhanced_embedding,
                    clinical_features['treatment']
                )

            # Strategy 5: Biomarker-Guided Attention
            if self.biomarker_attention is not None and 'biomarkers' in clinical_features:
                gene_encoding = encoding[:, 2:, :]  # [B, L, D]
                biomarker_guided_features = self.biomarker_attention(
                    gene_encoding,
                    clinical_features['biomarkers']
                )
                # Concatenate with concepts
                enhanced_embedding = torch.cat([enhanced_embedding, biomarker_guided_features], dim=1)

        # Main task prediction
        if self.ref_for_task:
            if biomarker_guided_features is not None:
                # Use enhanced embedding with biomarker features
                y = self.taskdecoder(enhanced_embedding)
            else:
                y = self.taskdecoder(enhanced_embedding)
        else:
            if biomarker_guided_features is not None:
                # Need to mask reference features
                mask_enhanced = torch.ones(enhanced_embedding.shape[1], dtype=torch.bool)
                mask_enhanced[self.ref_celltype_ids] = False
                emb_used_enhanced = enhanced_embedding[:, mask_enhanced]
                y = self.taskdecoder(emb_used_enhanced)
            else:
                y = self.taskdecoder(emb_used)

        gene_encoding = encoding[:, 2:, :]
        gene_ref = gene_encoding[:, self.ref_gene_ids, :]

        # Clinical outputs (optional)
        clinical_outputs = {}

        # Store concepts for clinical losses (use original embedding, not enhanced)
        clinical_outputs['concepts'] = embedding

        # Clinical feature integration outputs
        if clinical_features is not None:
            # Strategy 3: Pathway predictions (for pathway consistency loss)
            pathway_predictor = getattr(self, 'pathway_predictor', None)
            if pathway_predictor is not None:
                pathway_predictions = pathway_predictor(gene_encoding)
                clinical_outputs['pathway_predictions'] = pathway_predictions

            # Strategy 4: Auxiliary task predictions (TIDE, IPRES, phenotype)
            auxiliary_heads = getattr(self, 'auxiliary_heads', None)
            if auxiliary_heads is not None and (return_auxiliary or self.training):
                auxiliary_predictions = {}
                for task_name, aux_head in auxiliary_heads.items():
                    aux_pred = aux_head(embedding)
                    auxiliary_predictions[task_name] = aux_pred
                clinical_outputs['auxiliary_predictions'] = auxiliary_predictions

            # Store gene encoding for clinical losses
            clinical_outputs['gene_encoding'] = gene_encoding

        # Return with clinical outputs if any exist, otherwise backward compatible
        if clinical_outputs:
            return (embedding, (gene_ref, emb_ref)), y, clinical_outputs
        else:
            return (embedding, (gene_ref, emb_ref)), y