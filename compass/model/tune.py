# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:44:09 2023

@author: Wanxiang Shen
"""
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Torchdata
from tqdm import tqdm

tqdm.pandas(ascii=True)

from ..dataloader import GeneData
from .loss import (entropy_regularization, independence_loss,
                   reference_consistency_loss)


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)



def FT_Trainer(train_loader, model,
            optimizer, ssl_loss,
            tsk_loss, device,
            alpha=0.0,
            correction = 0.0,
            entropy_weight = 0.0,
            # NEW: Clinical feature integration losses
            concept_alignment_loss_fn=None,
            concept_alignment_loss_scale=0.0,
            concept_alignment_warmup_epochs=5,
            pathway_consistency_loss_fn=None,
            pathway_consistency_loss_scale=0.0,
            pathway_consistency_warmup_epochs=5,
            auxiliary_task_loss_fn=None,
            auxiliary_task_loss_scale=0.0,
            auxiliary_task_warmup_epochs=5,
            biomarker_names=None,  # For manual alignment mode
            current_epoch=0):


    model.train()
    total_loss = []
    total_ssl_loss = []
    total_tsk_loss = []
    # Track clinical loss components
    clinical_loss_components = {
        # Clinical feature integration losses
        'concept_alignment': [],
        'pathway_consistency': [],
        'auxiliary_task': []
    }

    #torch.autograd.set_detect_anomaly(True)
    #for data in tqdm(train_loader, ascii=True):
    for data in train_loader:

        # Unpack data (with optional metadata)
        if len(data) == 3:
            triplet, label, metadata = data
        else:
            triplet, label = data
            metadata = {}

        anchor_y_true, positive_y_true, negative_y_true = label
        anchor, positive, negative = triplet

        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        anchor_y_true = anchor_y_true.to(device)
        positive_y_true = positive_y_true.to(device)
        negative_y_true = negative_y_true.to(device)

        # Move metadata to device if present
        # Extract clinical features if present
        clinical_features = None
        if 'clinical_features' in metadata:
            clinical_features = metadata['clinical_features']
            # Move clinical features to device
            for key in clinical_features:
                if isinstance(clinical_features[key], torch.Tensor):
                    clinical_features[key] = clinical_features[key].to(device)

        optimizer.zero_grad()

        # Forward pass (with optional clinical features)
        anchor_output = model(anchor, epoch=current_epoch,
                            clinical_features=clinical_features)
        positive_output = model(positive, epoch=current_epoch,
                              clinical_features=clinical_features)
        negative_output = model(negative, epoch=current_epoch,
                              clinical_features=clinical_features)

        # Unpack outputs (backward compatible)
        if len(anchor_output) == 3:
            (anchor_emb, anchor_refg), anchor_y_pred, anchor_clinical = anchor_output
            (positive_emb, positive_refg), positive_y_pred, _ = positive_output
            (negative_emb, negative_refg), negative_y_pred, _ = negative_output
        else:
            (anchor_emb, anchor_refg), anchor_y_pred = anchor_output
            (positive_emb, positive_refg), positive_y_pred = positive_output
            (negative_emb, negative_refg), negative_y_pred = negative_output
            anchor_clinical = {}

        lss = ssl_loss(anchor_emb, positive_emb, negative_emb)

        ## remove batch effects by minimal the differences between house-keeping genes
        if correction != 0 :
            ref = reference_consistency_loss(anchor_refg[1], positive_refg[1], negative_refg[1]) 
            lss = (1-correction)*lss + correction*ref
            #print("Ref: {:.6f} - lss: {:.2f}".format(ref.item(), lss.item()))
        
        y_pred = anchor_y_pred #torch.cat([anchor_y_pred, positive_y_pred, negative_y_pred])
        y_true = anchor_y_true #torch.cat([anchor_y_true, positive_y_true, negative_y_true])
        tsk = tsk_loss(y_pred, y_true)

        if entropy_weight != 0 :
            entropy_reg  = entropy_regularization(y_pred)
            tsk = tsk * (1-entropy_weight) + entropy_reg * entropy_weight

        # NEW: Clinical losses
        loss_components = {}
        loss_components['ssl'] = (lss, (1 - alpha))
        loss_components['task'] = (tsk, alpha)

        # NEW: Concept Alignment Loss (Strategy 2)
        if concept_alignment_loss_scale > 0 and concept_alignment_loss_fn is not None and clinical_features is not None and current_epoch >= concept_alignment_warmup_epochs:
            # Extract cell type biomarkers from clinical features
            if 'cell_type' in clinical_features:
                cell_type_biomarkers = clinical_features['cell_type']  # [B, 62]

                concept_align_loss = concept_alignment_loss_fn(
                    concepts=anchor_clinical['concepts'],  # [B, 44]
                    biomarkers=cell_type_biomarkers,
                    biomarker_names=biomarker_names  # [B, 62]
                )
                loss_components['concept_alignment'] = (concept_align_loss, concept_alignment_loss_scale)

        # NEW: Pathway Consistency Loss (Strategy 3)
        if pathway_consistency_loss_scale > 0 and pathway_consistency_loss_fn is not None and clinical_features is not None and current_epoch >= pathway_consistency_warmup_epochs:
            # Extract pathway biomarkers from clinical features
            if 'pathway' in clinical_features:
                pathway_targets = clinical_features['pathway']  # [B, 42]

                # Extract gene encoding for pathway prediction
                with torch.no_grad():
                    encoding = model.inputencoder(anchor)  # [B, L, C]
                    gene_encoding = encoding[:, 2:, :]  # [B, 15672, C]

                pathway_loss = pathway_consistency_loss_fn(
                    gene_encoding=gene_encoding,
                    pathway_targets=pathway_targets
                )
                loss_components['pathway_consistency'] = (pathway_loss, pathway_consistency_loss_scale)

        # NEW: Auxiliary Task Loss (Strategy 4)
        if auxiliary_task_loss_scale > 0 and auxiliary_task_loss_fn is not None and clinical_features is not None and current_epoch >= auxiliary_task_warmup_epochs:
            # Build auxiliary targets dict
            auxiliary_targets = {}
            if 'auxiliary_TIDE' in clinical_features:
                auxiliary_targets['TIDE'] = clinical_features['auxiliary_TIDE']
            if 'auxiliary_IPRES' in clinical_features:
                auxiliary_targets['IPRES'] = clinical_features['auxiliary_IPRES']
            if 'auxiliary_phenotype' in clinical_features:
                auxiliary_targets['phenotype'] = clinical_features['auxiliary_phenotype']

            if auxiliary_targets:
                auxiliary_loss, auxiliary_preds = auxiliary_task_loss_fn(
                    concepts=anchor_clinical['concepts'],  # [B, 44]
                    targets_dict=auxiliary_targets
                )
                loss_components['auxiliary_task'] = (auxiliary_loss, auxiliary_task_loss_scale)

                # Store auxiliary predictions in clinical outputs for logging
                anchor_clinical['auxiliary_predictions'] = auxiliary_preds

        # Normalize scales and compute total loss
        total_scale = sum(scale for _, scale in loss_components.values())
        if total_scale > 0:
            loss = sum((component_loss * scale / total_scale) for component_loss, scale in loss_components.values())
        else:
            loss = (1 - alpha) * lss + alpha * tsk  # Fallback to original

        # Standard backward pass
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        total_ssl_loss.append(lss.item())
        total_tsk_loss.append(tsk.item())

    train_total_loss = np.mean(total_loss)
    train_ssl_loss = np.mean(total_ssl_loss)
    train_tsk_loss = np.mean(total_tsk_loss)

    clinical_components = {
        key: np.mean(values) if values else 0.0
        for key, values in clinical_loss_components.items()
    }

    return train_total_loss, train_ssl_loss, train_tsk_loss, clinical_components



@torch.no_grad()
def FT_Tester(test_loader, model, ssl_loss, tsk_loss,
           device, alpha=0., correction=0,
           entropy_weight=0.0,
           current_epoch=0):
    model.eval()
    total_loss = []
    total_ssl_loss = []
    total_tsk_loss = []

    for data in test_loader:
        # Unpack data (with optional metadata)
        if len(data) == 3:
            triplet, label, metadata = data
        else:
            triplet, label = data
            metadata = {}

        anchor, positive, negative = triplet
        anchor_y_true, positive_y_true, negative_y_true = label

        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        anchor_y_true = anchor_y_true.to(device)
        positive_y_true = positive_y_true.to(device)
        negative_y_true = negative_y_true.to(device)

        # Move metadata to device if present
        # Forward pass (with optional clinical outputs)
        anchor_output = model(anchor, epoch=current_epoch)
        positive_output = model(positive, epoch=current_epoch)
        negative_output = model(negative, epoch=current_epoch)
        
        # Unpack outputs (backward compatible)
        if len(anchor_output) == 3:
            (anchor_emb, anchor_refg), anchor_y_pred, anchor_clinical = anchor_output
            (positive_emb, positive_refg), positive_y_pred, _ = positive_output
            (negative_emb, negative_refg), negative_y_pred, _ = negative_output
        else:
            (anchor_emb, anchor_refg), anchor_y_pred = anchor_output
            (positive_emb, positive_refg), positive_y_pred = positive_output
            (negative_emb, negative_refg), negative_y_pred = negative_output
            anchor_clinical = {}

        lss = ssl_loss(anchor_emb, positive_emb, negative_emb)

        if correction != 0:
            ref = reference_consistency_loss(anchor_refg[1], positive_refg[1], negative_refg[1])
            lss = (1-correction)*lss + correction*ref

        y_pred = anchor_y_pred
        y_true = anchor_y_true
        tsk = tsk_loss(y_pred, y_true)

        if entropy_weight != 0:
            entropy_reg = entropy_regularization(y_pred)
            tsk = tsk * (1-entropy_weight) + entropy_reg * entropy_weight

        # Clinical losses (same as FT_Trainer but without backward pass)
        loss_components = {}
        loss_components['ssl'] = (lss, (1 - alpha))
        loss_components['task'] = (tsk, alpha)

        # Normalize scales and compute total loss
        total_scale = sum(scale for _, scale in loss_components.values())
        if total_scale > 0:
            loss = sum((component_loss * scale / total_scale) for component_loss, scale in loss_components.values())
        else:
            loss = (1 - alpha) * lss + alpha * tsk  # Fallback to original

        total_loss.append(loss.item())
        total_ssl_loss.append(lss.item())
        total_tsk_loss.append(tsk.item())

    test_total_loss = np.mean(total_loss)
    test_ssl_loss = np.mean(total_ssl_loss)
    test_tsk_loss = np.mean(total_tsk_loss)

    return test_total_loss, test_ssl_loss, test_tsk_loss




from sklearn.metrics import accuracy_score
from sklearn.metrics import auc as prc_auc_score
from sklearn.metrics import (confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score)


def scorer(y_true, y_pred):
    
    y_prob = y_pred[:, 1]
    y_pred = y_pred.argmax(axis=1)
    y_true = y_true.argmax(axis=1)
    if len(np.unique(y_true)) == 1:
        roc = np.nan
    else:
        roc = roc_auc_score(y_true, y_prob)
    _precision, _recall, _ = precision_recall_curve(y_true, y_prob)
    prc = prc_auc_score(_recall, _precision)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    mcc  = matthews_corrcoef(y_true, y_pred)
    
    return f1, mcc, prc, roc, acc


@torch.no_grad()
def Evaluator(test_loader, model, device):
    model.eval()
    y_trues = []
    y_preds = []
    for data in test_loader:
        # Handle optional metadata
        if len(data) == 3:
            triplet, label, metadata = data
        else:
            triplet, label = data
        anchor, positive, negative = triplet
        anchor_y_true, positive_y_true, negative_y_true = label
        
        anchor = anchor.to(device)
        anchor_y_true = anchor_y_true.to(device)

        # Extract clinical features if available from metadata
        clinical_features = None
        if len(data) == 3 and 'clinical_features' in metadata:
            clinical_features = metadata['clinical_features']
            if clinical_features is not None:
                # Move clinical features to device
                for key, value in clinical_features.items():
                    if torch.is_tensor(value):
                        clinical_features[key] = value.to(device)

        # Model may return clinical_outputs as third element
        model_output = model(anchor, clinical_features=clinical_features)
        if len(model_output) == 3:
            (anchor_emb, anchor_refg), anchor_y_pred, _ = model_output
        else:
            (anchor_emb, anchor_refg), anchor_y_pred = model_output
        
        y_trues.append(anchor_y_true)
        y_preds.append(anchor_y_pred)

    y_true =  torch.concat(y_trues, axis=0).cpu().detach().numpy()
    y_pred =  torch.concat(y_preds, axis=0).cpu().detach().numpy()

    f1, mcc, prc,roc, acc = scorer(y_true, y_pred)
    return f1, mcc, prc, roc, acc


    

@torch.no_grad()
def Predictor(dfcx, model, scaler, device = 'cpu', batch_size=512,  num_workers=4, clinical_features=None):
    """
    Make predictions with optional clinical features.

    Args:
        dfcx: Gene expression dataframe
        model: COMPASS model
        scaler: Data scaler
        device: Device to run on
        batch_size: Batch size for inference
        num_workers: Number of data loader workers
        clinical_features: Optional dict of clinical features (from ClinicalFeatureLoader.transform())
                          Keys: 'treatment', 'cell_type', 'pathway', 'auxiliary_TIDE', etc.
                          Values: numpy arrays [N, feature_dim]

    Returns:
        Tuple of (embeddings_df, predictions_df)
    """
    model.eval()
    dfcx = scaler.transform(dfcx)

    predict_tcga = GeneData(dfcx, clinical_features=clinical_features)
    predict_loader = Torchdata.DataLoader(predict_tcga,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          worker_init_fn = worker_init_fn,
                                          num_workers=num_workers,
                                          persistent_workers=True if num_workers >0 else False)
    embds = []
    ys = []
    for data in tqdm(predict_loader, ascii=True):
        # Unpack data (with optional metadata for clinical features)
        # Check if data is a tuple with metadata (anchor, metadata_dict)
        if isinstance(data, list) and len(data) == 2 and isinstance(data[1], dict):
            anchor, metadata = data
            # Move clinical features to device if present
            clinical_batch = None
            if 'clinical_features' in metadata:
                clinical_batch = metadata['clinical_features']
                for key in clinical_batch:
                    if isinstance(clinical_batch[key], torch.Tensor):
                        clinical_batch[key] = clinical_batch[key].to(device)
        else:
            # No metadata - just the anchor tensor
            anchor = data
            clinical_batch = None

        anchor = anchor.to(device)

        # Forward pass with clinical features
        output = model(anchor, clinical_features=clinical_batch)

        # Unpack outputs (backward compatible)
        if len(output) == 3:
            (anchor_emb, anchor_refg), anchor_ys, _ = output
        else:
            (anchor_emb, anchor_refg), anchor_ys = output

        embds.append(anchor_emb)
        ys.append(anchor_ys)


    embeddings  = torch.concat(embds, axis=0).cpu().detach().numpy()
    predictions = torch.concat(ys, axis=0).cpu().detach().numpy()

    if len(model.embed_feature_names) == embeddings.shape[1]:
        columns = model.embed_feature_names
    else:
        columns = model.proj_feature_names

    dfe = pd.DataFrame(embeddings, index = predict_tcga.patient_name,
                       columns = columns)

    ref_cols = model.latentprojector.GENESET.iloc[model.latentprojector.CELLPATHWAY.loc['Reference']].index.tolist()
    ref_cols.append('Reference')

    if not model.ref_for_task:
        dfe = dfe[dfe.columns.difference(ref_cols)]

    dfp = pd.DataFrame(predictions, index = predict_tcga.patient_name)

    return dfe, dfp




@torch.no_grad()
def Extractor(dfcx, model, scaler, device = 'cpu', batch_size = 512,  num_workers=4, with_gene_level = False):
    '''
    Extract geneset-level and celltype-level features
    '''
    model.eval()
    dfcx = scaler.transform(dfcx)
    genesetprojector = model.latentprojector.genesetprojector
    cellpathwayprojector = model.latentprojector.cellpathwayprojector

    predict_tcga = GeneData(dfcx)
    predict_loader = Torchdata.DataLoader(predict_tcga, 
                                          batch_size=batch_size, 
                                          shuffle=False,
                                          pin_memory=True, 
                                          worker_init_fn = worker_init_fn,
                                          num_workers=num_workers)
    geneset_feat = []
    celltype_feat = []
    gene_feat = []
    
    for anchor in tqdm(predict_loader, ascii=True):
        anchor = anchor.to(device)
        encoding = model.inputencoder(anchor)

        
        gene_level_proj = genesetprojector.geneset_scorer(encoding)[:,2:] #remove pid, cancer
        geneset_level_proj, cellpathway_level_proj = model.latentprojector(encoding)
    
        geneset_feat.append(geneset_level_proj)
        celltype_feat.append(cellpathway_level_proj)
        gene_feat.append(gene_level_proj)

    genefeatures = torch.concat(gene_feat, axis=0).cpu().detach().numpy() 
    genesetfeatures  = torch.concat(geneset_feat, axis=0).cpu().detach().numpy()
    celltypefeatures = torch.concat(celltype_feat, axis=0).cpu().detach().numpy()

    dfgeneset = pd.DataFrame(genesetfeatures, index = predict_tcga.patient_name, 
                             columns = model.geneset_feature_name)
    dfcelltype = pd.DataFrame(celltypefeatures, index = predict_tcga.patient_name, 
                              columns = model.celltype_feature_name)

    dfgene = pd.DataFrame(genefeatures, index = predict_tcga.patient_name, 
                          columns = predict_tcga.feature_name)

    return dfgene, dfgeneset, dfcelltype






@torch.no_grad()
def Projector(dfcx, model, scaler, device = 'cpu', batch_size = 64,  num_workers=4):
    '''
    Extract geneset-level and celltype-level features
    '''

    model.eval()
    gs_projector = model.latentprojector.genesetprojector
    ct_projector = model.latentprojector.cellpathwayprojector
    
    predict_tcga = GeneData(dfcx)
    predict_loader = Torchdata.DataLoader(predict_tcga, 
                                          batch_size=batch_size, 
                                          shuffle=False,
                                          pin_memory=True, 
                                          worker_init_fn = worker_init_fn,
                                          num_workers=num_workers)
    geneset_feat = []
    celltype_feat = []
    
    for anchor in tqdm(predict_loader, ascii=True):
        anchor = anchor.to(device)
    
        x = model.inputencoder(anchor)
        pid_encoding = x[:, 0:1, :]  # take the learnbale patient id token 
        cancer_encoding = x[:, 1:2, :] # take the cancer_type token 
        gene_encoding = x[:, 2:, :] # take the gene encoding 
        
        geneset_level_proj = gs_projector.geneset_aggregator(x)
        
        b,f,c = geneset_level_proj.shape
        
        ct_feats = []
        for i in range(c):
            ct_ = ct_projector(geneset_level_proj[:,:,i])
            ct_ = ct_.cpu().detach().numpy()
            ct_feats.append(ct_)
            
        celltype_level_proj = np.stack(ct_feats, axis=-1)
        geneset_level_proj = geneset_level_proj.cpu().detach().numpy()
        
        geneset_feat.append(geneset_level_proj)
        celltype_feat.append(celltype_level_proj)
    
    
    gs_feat = np.concatenate(geneset_feat, axis=0)
    ct_feat = np.concatenate(celltype_feat, axis=0)
    
    b,f,c = gs_feat.shape
    feature_name = model.latentprojector.GENESET.index
    feature_sample_labels = [dfcx.index[i//f] + '$$' + feature_name[i%f] for i in range(b*f)]
    dfgs = pd.DataFrame(gs_feat.reshape(b*f, c), 
                        index = feature_sample_labels, 
                        columns = ['channel_%s' % i for i in range(c)])
    
    
    b,f,c = ct_feat.shape
    feature_name = model.latentprojector.CELLPATHWAY.index
    index = [dfcx.index[i//f] + '$$' + feature_name[i%f] for i in range(b*f)]
    columns = ['channel_%s' % i for i in range(c)]
    dfct = pd.DataFrame(ct_feat.reshape(b*f, c), 
                        index = index, 
                        columns = columns)
    return dfgs, dfct