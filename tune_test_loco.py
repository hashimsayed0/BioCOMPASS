import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

sns.set_theme(style='white', font_scale=1.5)

import os
import sys

# Add parent directory to path to import compass module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compass import FineTuner, loadcompass


def str2bool(v):
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Load cancer type to code mapping
CANCER_CODE_PATH = 'compass/tokenizer/cancer_code.json'
with open(CANCER_CODE_PATH, 'r') as f:
    CANCER_TYPE_TO_CODE = json.load(f)

# Function to discover cohorts from sample info file
def discover_cohorts(sample_info_file):
    """Load sample info and discover all cohorts with their cancer types."""
    df_info = pd.read_csv(sample_info_file, sep='\t', index_col='sample_name')

    # Filter for pre-treatment samples with valid Responder (exclude "na_responder")
    valid_mask = (
        (df_info['Responder'] != 'na_responder') &
        (df_info['Responder'].notna()) &
        (df_info['Sample_Treatment'] == 'pre_sample_treatment')
    )
    df_info_filtered = df_info[valid_mask]
    print(f"Filtered to {len(df_info_filtered)} pre-treatment samples with valid responder status")

    # Get unique cohorts and their cancer types
    cohort_cancer_mapping = df_info_filtered.groupby('Dataset')['TCGA_Study'].unique()

    # Create mapping (assuming each cohort has only one cancer type)
    cohort_to_cancer = {}
    valid_cohorts = []

    for cohort, tissues in cohort_cancer_mapping.items():
        # Use the most common tissue type if multiple exist, excluding 'na_tcga_study'
        tissue_counts = df_info_filtered[df_info_filtered['Dataset'] == cohort]['TCGA_Study'].value_counts()

        # Skip cohorts with no samples
        if len(tissue_counts) == 0:
            print(f"Warning: Cohort '{cohort}' has no samples with valid TCGA_Study, skipping...")
            continue

        # Filter out 'na_tcga_study' if there are other valid options
        valid_tissues = tissue_counts[tissue_counts.index != 'na_tcga_study']
        if len(valid_tissues) > 0:
            cohort_to_cancer[cohort] = valid_tissues.index[0]
            valid_cohorts.append(cohort)
        else:
            # If all are 'na_tcga_study', use it but warn
            print(f"Warning: Cohort '{cohort}' only has 'na_tcga_study' as cancer type")
            cohort_to_cancer[cohort] = tissue_counts.index[0]
            valid_cohorts.append(cohort)

    all_cohorts = sorted(valid_cohorts)

    return all_cohorts, cohort_to_cancer, df_info_filtered

def get_cancer_code(cohort_name, cohort_to_cancer):
    """Get cancer code for a given cohort."""
    cancer_type = cohort_to_cancer.get(cohort_name)
    if cancer_type is None:
        raise ValueError(f"Unknown cohort: {cohort_name}")

    cancer_code = CANCER_TYPE_TO_CODE.get(cancer_type)
    if cancer_code is None:
        raise ValueError(f"Cancer type {cancer_type} not found in cancer_code.json")

    return cancer_code

def train_and_test_loco(test_cohort, all_cohorts, cohort_to_cancer, df_tpm_full,
                        df_sample_info_filtered, labels_full, model, expected_genes,
                        args, wandb_group, val_ratio,
                        output_dir='example/model', results_dir='example/results'):
    """Train and test for a single leave-one-cohort-out fold."""

    print("\n" + "="*80)
    print(f"COMPASS FINE-TUNING: LEAVE-ONE-COHORT-OUT (iAtlas - Test: {test_cohort})")
    print("="*80)

    # Training cohorts: all except test_cohort
    train_cohorts = [c for c in all_cohorts if c != test_cohort]

    print(f"\nTraining cohorts: {train_cohorts}")
    print(f"Test cohort: {test_cohort}\n")

    # Load and combine training data from all cohorts except test cohort
    print(f"Loading training data from cohorts: {train_cohorts}...")
    train_dfs = []
    train_labels_list = []
    sample_to_cohort = {}  # Track which cohort each sample belongs to

    for cohort in train_cohorts:
        cohort_mask = df_sample_info_filtered['Dataset'] == cohort
        cohort_samples = df_sample_info_filtered[cohort_mask].index

        if len(cohort_samples) > 0:
            cohort_cancer_type = cohort_to_cancer[cohort]
            cohort_cancer_code = get_cancer_code(cohort, cohort_to_cancer)
            print(f"  Loading {cohort} ({cohort_cancer_type}, code={cohort_cancer_code})...")

            # Get TPM data for this cohort
            df_tpm_cohort = df_tpm_full.loc[cohort_samples]

            # Get labels for this cohort
            labels_cohort = labels_full.loc[cohort_samples]

            # Track cohort for each sample
            for sample_id in df_tpm_cohort.index:
                sample_to_cohort[sample_id] = cohort

            train_dfs.append(df_tpm_cohort)
            train_labels_list.append(labels_cohort)

            n_responders = np.sum(labels_cohort == 1)
            n_non_responders = np.sum(labels_cohort == 0)
            print(f"    Samples: {len(df_tpm_cohort)}, Response: {n_responders}, No Response: {n_non_responders}")
        else:
            print(f"    Warning: No samples for {cohort}")

    # Combine all training data
    print("\nCombining training data...")
    df_train = pd.concat(train_dfs, axis=0)
    train_labels = pd.concat(train_labels_list, axis=0)

    print(f"Total training samples: {len(df_train)}")
    print(f"Training Response: {np.sum(train_labels == 1)}")
    print(f"Training No Response: {np.sum(train_labels == 0)}")

    # Print cancer type distribution in training data
    print("\nCancer type distribution in training data:")
    cancer_type_counts = {}
    for sample_id in df_train.index:
        cohort = sample_to_cohort[sample_id]
        cancer_type = cohort_to_cancer[cohort]
        cancer_type_counts[cancer_type] = cancer_type_counts.get(cancer_type, 0) + 1
    for cancer_type, count in sorted(cancer_type_counts.items()):
        cancer_code = CANCER_TYPE_TO_CODE[cancer_type]
        print(f"  {cancer_type} (code={cancer_code}): {count} samples")

    # Load test data
    test_cancer_type = cohort_to_cancer[test_cohort]
    test_cancer_code = get_cancer_code(test_cohort, cohort_to_cancer)
    print(f"\nLoading test data from {test_cohort} ({test_cancer_type}, code={test_cancer_code})...")

    test_mask = df_sample_info_filtered['Dataset'] == test_cohort
    test_samples = df_sample_info_filtered[test_mask].index

    df_test_tpm = df_tpm_full.loc[test_samples]
    test_labels = labels_full.loc[test_samples]

    print(f"Test samples: {len(df_test_tpm)}")
    print(f"Test Response: {np.sum(test_labels == 1)}")
    print(f"Test No Response: {np.sum(test_labels == 0)}")

    # Align training data with model expectations
    print("\nAligning training data with model expectations...")
    common_genes_train = list(set(df_train.columns) & set(expected_genes))
    print(f"Found {len(common_genes_train)} common genes in training data")

    df_train_aligned = df_train[common_genes_train].reindex(columns=expected_genes, fill_value=0)

    # Handle missing values in aligned training data
    if df_train_aligned.isnull().any().any():
        nan_count = df_train_aligned.isnull().sum().sum()
        print(f"Found {nan_count} missing values in aligned training data, filling with 0")
        df_train_aligned = df_train_aligned.fillna(0)

    # Add cancer_code column based on each sample's cohort
    # This must be the first column for COMPASS
    print("\nAssigning cancer codes to training samples...")
    train_cancer_codes = []
    for sample_id in df_train_aligned.index:
        cohort = sample_to_cohort[sample_id]
        cancer_code = get_cancer_code(cohort, cohort_to_cancer)
        train_cancer_codes.append(cancer_code)

    df_train_aligned.insert(0, 'cancer_code', train_cancer_codes)
    print(f"Cancer codes assigned: {set(train_cancer_codes)}")

    # Convert labels to one-hot encoding
    print("\nPreparing labels for fine-tuning...")
    train_labels_df = pd.DataFrame({
        0: (train_labels == 0).astype(int),
        1: (train_labels == 1).astype(int)
    }, index=train_labels.index)

    print(f"Training data shape: {df_train_aligned.shape}")
    print(f"Training labels shape: {train_labels_df.shape}")

    # Align test data with model expectations
    print("\nAligning test data with model expectations...")
    common_genes_test = list(set(df_test_tpm.columns) & set(expected_genes))
    print(f"Found {len(common_genes_test)} common genes in test data")

    df_test_aligned = df_test_tpm[common_genes_test].reindex(columns=expected_genes, fill_value=0)

    # Handle missing values in aligned test data
    if df_test_aligned.isnull().any().any():
        nan_count = df_test_aligned.isnull().sum().sum()
        print(f"Found {nan_count} missing values in aligned test data, filling with 0")
        df_test_aligned = df_test_aligned.fillna(0)

    # Add cancer_code column for test cohort
    print(f"Assigning cancer code {test_cancer_code} ({test_cancer_type}) to test samples...")
    df_test_aligned.insert(0, 'cancer_code', test_cancer_code)

    # Convert test labels to one-hot encoding
    test_labels_df = pd.DataFrame({
        0: (test_labels == 0).astype(int),
        1: (test_labels == 1).astype(int)
    }, index=test_labels.index)

    print(f"Test data shape: {df_test_aligned.shape}")
    print(f"Test labels shape: {test_labels_df.shape}")

    # Split into train and validation if val_ratio > 0
    if val_ratio > 0:
        from sklearn.model_selection import train_test_split

        print(f"\n{'='*80}")
        print(f"CREATING VALIDATION SPLIT (ratio={val_ratio})")
        print(f"{'='*80}")

        # Create combined stratification labels (cohort_response)
        stratify_labels = [f"{sample_to_cohort[idx]}_{int(label)}"
                           for idx, label in zip(df_train_aligned.index, train_labels)]

        try:
            # Attempt stratified split
            train_indices, val_indices = train_test_split(
                range(len(df_train_aligned)),
                test_size=val_ratio,
                stratify=stratify_labels,
                random_state=args.seed
            )
        except ValueError as e:
            # Fall back to response-only stratification if cohort+response fails
            print(f"Warning: Stratification by cohort+response failed ({e})")
            print("Falling back to response-only stratification...")
            train_indices, val_indices = train_test_split(
                range(len(df_train_aligned)),
                test_size=val_ratio,
                stratify=train_labels.values,
                random_state=args.seed
            )

        # Split the data
        df_train_split = df_train_aligned.iloc[train_indices]
        df_val_aligned = df_train_aligned.iloc[val_indices]
        train_labels_split = train_labels.iloc[train_indices]
        val_labels = train_labels.iloc[val_indices]

        # Convert validation labels to one-hot
        val_labels_df = pd.DataFrame({
            0: (val_labels == 0).astype(int),
            1: (val_labels == 1).astype(int)
        }, index=val_labels.index)

        # Convert training labels to one-hot
        train_labels_split_df = pd.DataFrame({
            0: (train_labels_split == 0).astype(int),
            1: (train_labels_split == 1).astype(int)
        }, index=train_labels_split.index)

        # Print validation statistics
        print(f"\nTraining set: {len(df_train_split)} samples")
        print(f"  Response: {np.sum(train_labels_split == 1)}")
        print(f"  No Response: {np.sum(train_labels_split == 0)}")

        print(f"\nValidation set: {len(df_val_aligned)} samples")
        print(f"  Response: {np.sum(val_labels == 1)}")
        print(f"  No Response: {np.sum(val_labels == 0)}")

        # Print cohort distribution in validation
        val_cohort_dist = {}
        for idx in df_val_aligned.index:
            cohort = sample_to_cohort[idx]
            val_cohort_dist[cohort] = val_cohort_dist.get(cohort, 0) + 1
        print(f"\nValidation cohort distribution:")
        for cohort, count in sorted(val_cohort_dist.items()):
            print(f"  {cohort}: {count} samples")
        print(f"{'='*80}\n")
    else:
        # No validation split - use all training data
        df_train_split = df_train_aligned
        train_labels_split_df = train_labels_df
        df_val_aligned = None
        val_labels_df = None
    # Fine-tune the model with training data
    print("\n" + "="*80)
    print("FINE-TUNING COMPASS MODEL")
    print("="*80)

    ft_args = {
        'mode': args.mode,  # Fine-tuning mode: FFT (Full), PFT (Partial), LFT (Linear)
        'lr': args.lr,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'max_epochs': args.max_epochs,
        'patience': args.patience,
        'seed': args.seed,
        'load_decoder': False,  # False because we're loading a pre-trained model, not a fine-tuned one
        'with_wandb': args.with_wandb,
        'wandb_project': args.wandb_project,
        'wandb_entity': args.wandb_entity,
        'wandb_dir': args.wandb_dir,
        'wandb_group': wandb_group,
        'wandb_name': f"loco_{test_cohort}",
        'wandb_config': {
            'mode': args.mode,  # Fine-tuning mode: FFT (Full), PFT (Partial), LFT (Linear)
            'lr': args.lr,
            'batch_size': args.batch_size,
            'max_epochs': args.max_epochs,
            'patience': args.patience,
            'seed': args.seed,
            'load_decoder': False,
            'test_cohort': test_cohort,
            'train_cohorts': [c for c in all_cohorts if c != test_cohort],
            'num_cohorts': len(all_cohorts),
            'val_ratio': val_ratio,
            # Clinical feature integration configuration
            'clinical_features_file': args.clinical_features_file,
            'treatment_gating_enabled': args.treatment_gating_enabled,
            'concept_alignment_loss_scale': args.concept_alignment_loss_scale,
            'concept_alignment_mode': args.concept_alignment_mode,
            'pathway_consistency_loss_scale': args.pathway_consistency_loss_scale,
            'auxiliary_task_loss_scale': args.auxiliary_task_loss_scale,
            'biomarker_attention_enabled': args.biomarker_attention_enabled,
        },
        'verbose': False,

        # Clinical feature integration parameters
        'clinical_features_file': args.clinical_features_file,
        'treatment_gating_enabled': args.treatment_gating_enabled,
        'treatment_gating_hidden_dim': args.treatment_gating_hidden_dim,
        'concept_alignment_loss_scale': args.concept_alignment_loss_scale,
        'concept_alignment_mode': args.concept_alignment_mode,
        'concept_alignment_warmup_epochs': args.concept_alignment_warmup_epochs,
        'pathway_consistency_loss_scale': args.pathway_consistency_loss_scale,
        'pathway_consistency_warmup_epochs': args.pathway_consistency_warmup_epochs,
        'auxiliary_task_loss_scale': args.auxiliary_task_loss_scale,
        'auxiliary_task_warmup_epochs': args.auxiliary_task_warmup_epochs,
        'biomarker_attention_enabled': args.biomarker_attention_enabled,
        'biomarker_attention_dim': args.biomarker_attention_dim,
        'biomarker_attention_heads': args.biomarker_attention_heads,
    }

    finetuner = FineTuner(model, **ft_args)

    # Fine-tune the model using training data
    print("Starting fine-tuning...")
    if val_ratio == 0:
        # No validation - use tune()
        finetuner.tune(
            dfcx_train=df_train_aligned,
            dfy_train=train_labels_df,
            min_mcc=0.8
        )
    else:
        # With validation - use tune_with_test()
        finetuner.tune_with_test(
            dfcx_train=df_train_split,
            dfy_train=train_labels_split_df,
            dfcx_test=df_val_aligned,
            dfy_test=val_labels_df,
            min_mcc=0.8
        )

    # Evaluate on test data
    print("\n" + "="*80)
    print(f"EVALUATING ON TEST COHORT ({test_cohort})")
    print("="*80)

    # Make predictions using the fine-tuned model
    print("Making predictions on test data...")
    # Clinical features will be automatically loaded from clinical_feature_loader if it was configured
    _, df_pred = finetuner.predict(df_test_aligned, batch_size=128, num_workers=args.num_workers)

    # Extract prediction scores
    if df_pred.shape[1] == 2:
        predictions = df_pred.iloc[:, 1].values  # Use column 1 for positive class (Response)
    else:
        predictions = df_pred.values.flatten()

    # Calculate metrics
    print("\n" + "="*80)
    print(f"COMPASS EVALUATION RESULTS ON {test_cohort} DATASET")
    print("="*80)
    print(f"Dataset: {test_cohort}")
    print(f"Samples: {len(df_test_aligned)}")
    print(f"Responders: {np.sum(test_labels == 1)}")
    print(f"Non-responders: {np.sum(test_labels == 0)}")
    print()

    # Calculate AUC
    auc = roc_auc_score(test_labels, predictions)

    # Binary predictions using 0.5 threshold
    pred_binary = (predictions > 0.5).astype(int)

    # Calculate classification metrics
    accuracy = accuracy_score(test_labels, pred_binary)
    precision = precision_score(test_labels, pred_binary, zero_division=0)
    recall = recall_score(test_labels, pred_binary, zero_division=0)
    f1 = f1_score(test_labels, pred_binary, zero_division=0)

    # Display results
    print("Performance Metrics:")
    print(f"  AUC:       {auc:.3f}")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-score:  {f1:.3f}")
    print()

    # Prediction statistics
    print("Prediction Statistics:")
    print(f"  Mean prediction score: {np.mean(predictions):.3f}")
    print(f"  Std prediction score:  {np.std(predictions):.3f}")
    print(f"  Min prediction score:  {np.min(predictions):.3f}")
    print(f"  Max prediction score:  {np.max(predictions):.3f}")
    print("="*80)

    # Log test metrics to W&B if enabled
    if args.with_wandb:
        test_metrics = {
            'test_auc': auc,
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'test_mean_pred': np.mean(predictions),
            'test_std_pred': np.std(predictions),
            'test_n_samples': len(df_test_aligned),
            'test_n_responders': np.sum(test_labels == 1),
            'test_n_non_responders': np.sum(test_labels == 0)
        }
        finetuner.log_test_metrics(test_metrics)

    # Save detailed results
    results_df = pd.DataFrame({
        'sample_id': df_test_aligned.index,
        'true_label': test_labels.values,
        'predicted_score': predictions,
        'predicted_binary': pred_binary
    })
    cohort_short = test_cohort.replace('_', '').lower()
    results_file = os.path.join(results_dir, f'iatlas_{cohort_short}_pft_predictions.csv')
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")

    # Finish W&B run for this fold
    if args.with_wandb:
        finetuner.finish_wandb()

    # Return metrics for summary
    return {
        'test_cohort': test_cohort,
        'n_test_samples': len(df_test_aligned),
        'n_train_samples': len(df_train_aligned),
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_pred': np.mean(predictions),
        'std_pred': np.std(predictions)
    }


# Parse command line arguments
parser = argparse.ArgumentParser(description='Fine-tune COMPASS model with leave-one-cohort-out cross-validation on iAtlas')
parser.add_argument('--cohorts', type=str,
                    default='Gide_Cell_2019,HugoLo_IPRES_2016,IMmotion150,IMVigor210,Kim_NatMed_2018,Liu_NatMed_2019,Riaz_Nivolumab_2017,VanAllen_antiCTLA4_2015',
                    help='Comma-separated list of cohorts to include in LOCO (default: 8 main cohorts)')
parser.add_argument('--test_cohort', type=str, default=None,
                    help='Single cohort to test. If specified, runs only this cohort.')
parser.add_argument('--gene_exp_file', type=str,
                    default='data/gene_exp.tsv',
                    help='Path to gene expression data file')
parser.add_argument('--labels_file', type=str,
                    default='data/labels.tsv',
                    help='Path to sample info file')
parser.add_argument('--model_path', type=str, default='models/pretrainer.pt',
                    help='Path to pre-trained model')
parser.add_argument('--output_dir', type=str, default='models',
                    help='Directory to save fine-tuned models (default: models)')
parser.add_argument('--results_dir', type=str, default='results',
                    help='Directory to save prediction results (default: results)')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size for fine-tuning (default: 16)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate (default: 1e-3)')
parser.add_argument('--max_epochs', type=int, default=100,
                    help='Maximum epochs for fine-tuning (default: 100)')
parser.add_argument('--patience', type=int, default=10,
                    help='Early stopping patience (default: 10)')
parser.add_argument('--with_wandb', type=str2bool, default=True,
                    help='Enable Weights & Biases logging (default: True)')
parser.add_argument('--wandb_project', type=str, default='biocompass',
                    help='W&B project name (default: biocompass)')
parser.add_argument('--wandb_entity', type=str, default=None,
                    help='W&B entity/username (default: None)')
parser.add_argument('--wandb_dir', type=str, default='./wandb_logs',
                    help='W&B logging directory (default: ./wandb_logs)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')
parser.add_argument('--num_workers', type=int, default=8,
                    help='Number of DataLoader workers (default: 8)')
parser.add_argument('--mode', type=str, default='PFT',
                    choices=['FFT', 'PFT', 'LFT'],
                    help='Fine-tuning mode: FFT (Full), PFT (Partial), LFT (Linear) (default: PFT)')
parser.add_argument('--val_ratio', type=float, default=0.0,
                    help='Validation set ratio (0.0-1.0). If 0, no validation set. (default: 0.0)')


# Clinical feature integration parameters
parser.add_argument('--clinical_features_file', type=str, default=None,
                    help='Path to clinical features TSV file')
parser.add_argument('--treatment_gating_enabled', type=str2bool, default=False,
                    help='Enable treatment-aware gating (Strategy 1) (default: False)')
parser.add_argument('--treatment_gating_hidden_dim', type=int, default=32,
                    help='Hidden dimension for treatment gating network (default: 32)')
parser.add_argument('--concept_alignment_loss_scale', type=float, default=0.0,
                    help='Scale for concept alignment loss (Strategy 2) (default: 0.0)')
parser.add_argument('--concept_alignment_mode', type=str, default='manual',
                    choices=['manual', 'correlation', 'learnable'],
                    help='Concept alignment mode (default: manual)')
parser.add_argument('--concept_alignment_warmup_epochs', type=int, default=5,
                    help='Epochs before enabling concept alignment loss (default: 5)')
parser.add_argument('--pathway_consistency_loss_scale', type=float, default=0.0,
                    help='Scale for pathway consistency loss (Strategy 3) (default: 0.0)')
parser.add_argument('--pathway_consistency_warmup_epochs', type=int, default=5,
                    help='Epochs before enabling pathway consistency loss (default: 5)')
parser.add_argument('--auxiliary_task_loss_scale', type=float, default=0.0,
                    help='Scale for auxiliary task loss (Strategy 4) (default: 0.0)')
parser.add_argument('--auxiliary_task_warmup_epochs', type=int, default=5,
                    help='Epochs before enabling auxiliary task loss (default: 5)')
parser.add_argument('--biomarker_attention_enabled', type=str2bool, default=False,
                    help='Enable biomarker-guided attention (Strategy 5) (default: False)')
parser.add_argument('--biomarker_attention_dim', type=int, default=32,
                    help='Dimension for biomarker attention (default: 32)')
parser.add_argument('--biomarker_attention_heads', type=int, default=4,
                    help='Number of attention heads for biomarker attention (default: 4)')

args = parser.parse_args()

# Validate val_ratio
if not (0.0 <= args.val_ratio < 1.0):
    raise ValueError(f"val_ratio must be between 0.0 and 1.0, got {args.val_ratio}")

# Set random seeds for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Discover cohorts from sample info file
print("Discovering cohorts from sample info file...")
discovered_cohorts, cohort_to_cancer, df_sample_info_filtered = discover_cohorts(args.labels_file)

# Parse cohorts from args (comma-separated string)
all_cohorts = [c.strip() for c in args.cohorts.split(',')]
# Validate that all specified cohorts exist in data
for cohort in all_cohorts:
    if cohort not in discovered_cohorts:
        raise ValueError(f"Cohort '{cohort}' not found in data. Available: {', '.join(discovered_cohorts)}")

print(f"Using {len(all_cohorts)} cohorts:")
for cohort in all_cohorts:
    cancer_type = cohort_to_cancer[cohort]
    n_samples = len(df_sample_info_filtered[df_sample_info_filtered['Dataset'] == cohort])
    print(f"  {cohort}: {cancer_type} ({n_samples} samples)")

# Select cohorts for LOCO
if args.test_cohort is not None:
    # Single cohort mode (for sweeps)
    if args.test_cohort not in all_cohorts:
        raise ValueError(f"Test cohort '{args.test_cohort}' not found in cohorts. Available: {', '.join(all_cohorts)}")
    cohorts_to_test = [args.test_cohort]
    all_cohorts_pool = all_cohorts
else:
    cohorts_to_test = all_cohorts

print("\n" + "="*80)
print("LEAVE-ONE-COHORT-OUT CROSS-VALIDATION")
print("="*80)
print(f"Number of folds: {len(cohorts_to_test)}")
print(f"Cohorts to test: {cohorts_to_test}")
print("="*80)

# Load TPM data (Run_ID as rows, genes as columns)
print("\nLoading gene expression data...")
print(f"  Gene expression file: {args.gene_exp_file}")
df_tpm_full = pd.read_csv(args.gene_exp_file, sep='\t', index_col='Run_ID')
print(f"  Gene expression data shape: {df_tpm_full.shape}")

# Align TPM with filtered samples
common_samples = df_tpm_full.index.intersection(df_sample_info_filtered.index)
df_tpm_full = df_tpm_full.loc[common_samples]
df_sample_info_filtered = df_sample_info_filtered.loc[common_samples]
print(f"  Aligned samples: {len(common_samples)}")

# Map Responder to binary labels ("true_responder" -> 1, "false_responder" -> 0)
responder_map = {'true_responder': 1, 'false_responder': 0}
labels_full = df_sample_info_filtered['Responder'].map(responder_map)

# Load pre-trained COMPASS model once
print("\nLoading pre-trained COMPASS model...")
print(f"Model path: {args.model_path}")
try:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_template = loadcompass(args.model_path, map_location=device)
    print(f"Model loaded successfully on {device}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Attempting to load on CPU...")
    model_template = loadcompass(args.model_path, map_location='cpu')
    device = 'cpu'
    print("Model loaded on CPU")

# Get the genes expected by the model
expected_genes = model_template.scaler.scaler.feature_names_in_
print(f"Model expects {len(expected_genes)} genes")

# Perform LOCO cross-validation
all_results = []

# Generate W&B group name based on timestamp and mode
import time
wandb_group = f"loco_{args.mode}_seed{args.seed}_{int(time.time())}"

# Determine the cohort pool for training (all cohorts except test cohort)
if args.test_cohort is not None:
    cohort_pool = all_cohorts_pool
else:
    cohort_pool = cohorts_to_test

for i, test_cohort in enumerate(cohorts_to_test):
    print(f"\n\n{'#'*80}")
    print(f"# FOLD {i+1}/{len(cohorts_to_test)}: Testing on {test_cohort}")
    print(f"{'#'*80}\n")

    # Clear CUDA cache between folds to prevent memory issues
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Load a fresh copy of the model for each fold
    if i > 0:
        print(f"Loading fresh model from {args.model_path}...")
        model = loadcompass(args.model_path, map_location=device)
    else:
        model = model_template

    # Train and test for this fold
    fold_results = train_and_test_loco(
        test_cohort=test_cohort,
        all_cohorts=cohort_pool,
        cohort_to_cancer=cohort_to_cancer,
        df_tpm_full=df_tpm_full,
        df_sample_info_filtered=df_sample_info_filtered,
        labels_full=labels_full,
        model=model,
        expected_genes=expected_genes,
        args=args,
        wandb_group=wandb_group,
        val_ratio=args.val_ratio,
        output_dir=args.output_dir,
        results_dir=args.results_dir
    )

    all_results.append(fold_results)

# Print summary of all folds
print("\n\n" + "="*80)
print("LEAVE-ONE-COHORT-OUT CROSS-VALIDATION SUMMARY")
print("="*80)
print()

# Create summary DataFrame
summary_df = pd.DataFrame(all_results)

# Display summary table
print("Per-Cohort Results:")
print("-" * 80)
print(f"{'Cohort':<20} {'Train N':>8} {'Test N':>8} {'AUC':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
print("-" * 80)
for _, row in summary_df.iterrows():
    print(f"{row['test_cohort']:<20} {row['n_train_samples']:>8} {row['n_test_samples']:>8} "
          f"{row['auc']:>8.3f} {row['accuracy']:>8.3f} {row['precision']:>8.3f} "
          f"{row['recall']:>8.3f} {row['f1']:>8.3f}")
print("-" * 80)

# Calculate and display overall statistics
print("\nOverall Statistics:")
print(f"  Mean AUC:       {summary_df['auc'].mean():.3f} ± {summary_df['auc'].std():.3f}")
print(f"  Mean Accuracy:  {summary_df['accuracy'].mean():.3f} ± {summary_df['accuracy'].std():.3f}")
print(f"  Mean Precision: {summary_df['precision'].mean():.3f} ± {summary_df['precision'].std():.3f}")
print(f"  Mean Recall:    {summary_df['recall'].mean():.3f} ± {summary_df['recall'].std():.3f}")
print(f"  Mean F1-score:  {summary_df['f1'].mean():.3f} ± {summary_df['f1'].std():.3f}")
print()
print(f"  Total test samples: {summary_df['n_test_samples'].sum()}")
print(f"  Cohorts tested: {len(summary_df)}")
print("="*80)

# Save summary results
summary_file = os.path.join(args.results_dir, 'iatlas_loco_summary.csv')
summary_df.to_csv(summary_file, index=False)
print(f"\nSummary results saved to: {summary_file}")

print("\nLOCO cross-validation completed successfully!")
