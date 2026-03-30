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


def _load_and_filter(clinical_features_file, all_cohorts=None):
    """Load clinical features, apply validity filter, and optionally restrict to given cohorts."""
    df_info = pd.read_csv(clinical_features_file, sep='\t', index_col='Run_ID')

    df_info_filtered = df_info[
        df_info['Responder'].notna() &
        (df_info['Sample_Treatment'] == 'Pre')
    ]
    print(f"Filtered to {len(df_info_filtered)} pre-treatment samples with valid responder status")

    if all_cohorts is not None:
        unknown = [c for c in all_cohorts if c not in df_info_filtered['Dataset'].unique()]
        if unknown:
            raise ValueError(f"Unknown cohorts in --all_cohorts: {unknown}. "
                             f"Available: {sorted(df_info_filtered['Dataset'].unique().tolist())}")
        df_info_filtered = df_info_filtered[df_info_filtered['Dataset'].isin(all_cohorts)]
        print(f"Cohort pre-filter applied ({len(all_cohorts)} cohorts): {all_cohorts}")
        print(f"Samples after cohort filter: {len(df_info_filtered)}")

    return df_info_filtered


def discover_cohorts(clinical_features_file, all_cohorts=None):
    """Load clinical features and discover all cohorts (LOCO)."""
    df_info_filtered = _load_and_filter(clinical_features_file, all_cohorts)
    all_groups = sorted(df_info_filtered['Dataset'].unique().tolist())
    group_assignments = df_info_filtered['Dataset']
    return all_groups, group_assignments, df_info_filtered


def discover_cancer_types(clinical_features_file, all_cohorts=None):
    """Discover unique TCGA cancer types and build group_assignments (LOCTO)."""
    df_info_filtered = _load_and_filter(clinical_features_file, all_cohorts)
    all_groups = sorted(df_info_filtered['TCGA_Study'].dropna().unique().tolist())
    group_assignments = df_info_filtered['TCGA_Study']
    return all_groups, group_assignments, df_info_filtered


def discover_ici_target_groups(clinical_features_file, all_cohorts=None):
    """Discover ICI target groups using ICI_Target from clinical features (LOTO).

    Groups: 'PD-1' (282), 'PD-L1' (463), 'CTLA4' (41), 'CTLA4 + PD1' (32).
    Samples with NaN ICI_Target are assigned 'unknown'.
    """
    df_info_filtered = _load_and_filter(clinical_features_file, all_cohorts)
    group_assignments = df_info_filtered['ICI_Target'].fillna('unknown')
    all_groups = sorted(group_assignments.unique().tolist())
    return all_groups, group_assignments, df_info_filtered


def train_and_test(test_group, all_groups, group_assignments,
                   sample_to_cancer_code, df_tpm_full, df_sample_info_filtered,
                   labels_full, model, expected_genes, args, wandb_group,
                   val_ratio, setting, output_dir='example/model',
                   results_dir='example/results'):
    """Train and test for a single leave-one-group-out fold.

    Works for all settings:
      loco  — group = cohort name (Dataset column)
      locto — group = cancer type string (TCGA_Study column)
      loto  — group = ICI_Target string (PD-1, PD-L1, CTLA4, CTLA4 + PD1)
    """

    setting_upper = setting.upper()
    print("\n" + "="*80)
    print(f"COMPASS FINE-TUNING: LEAVE-ONE-{setting_upper[2:]}-OUT (Test group: {test_group})")
    print("="*80)

    # Training groups: all except test_group
    train_groups = [g for g in all_groups if g != test_group]

    print(f"\nTraining groups: {train_groups}")
    print(f"Test group: {test_group}\n")

    # Identify test samples
    test_sample_mask = group_assignments == test_group
    test_samples = group_assignments[test_sample_mask].index

    # Identify training samples
    train_sample_mask = group_assignments.isin(train_groups)
    train_samples = group_assignments[train_sample_mask].index

    # Load and combine training data
    print(f"Loading training data ({len(train_samples)} samples)...")
    df_train = df_tpm_full.loc[train_samples]
    train_labels = labels_full.loc[train_samples]

    print(f"Total training samples: {len(df_train)}")
    print(f"Training Response: {np.sum(train_labels == 1)}")
    print(f"Training No Response: {np.sum(train_labels == 0)}")

    # Print cancer code distribution in training data
    print("\nCancer code distribution in training data:")
    code_counts = df_train.index.map(sample_to_cancer_code).value_counts().sort_index()
    for code, count in code_counts.items():
        print(f"  code={code}: {count} samples")

    # Load test data
    print(f"\nLoading test data from group '{test_group}' ({len(test_samples)} samples)...")
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

    if df_train_aligned.isnull().any().any():
        nan_count = df_train_aligned.isnull().sum().sum()
        print(f"Found {nan_count} missing values in aligned training data, filling with 0")
        df_train_aligned = df_train_aligned.fillna(0)

    # Add cancer_code column from clinical features TCGA_Study
    print("\nAssigning cancer codes to training samples...")
    train_cancer_codes = df_train_aligned.index.map(sample_to_cancer_code).tolist()
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

    if df_test_aligned.isnull().any().any():
        nan_count = df_test_aligned.isnull().sum().sum()
        print(f"Found {nan_count} missing values in aligned test data, filling with 0")
        df_test_aligned = df_test_aligned.fillna(0)

    # Add cancer_code column for test samples from clinical features TCGA_Study
    print(f"Assigning cancer codes to test samples...")
    test_cancer_codes = df_test_aligned.index.map(sample_to_cancer_code).tolist()
    df_test_aligned.insert(0, 'cancer_code', test_cancer_codes)

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

        # Stratify by cohort+response
        stratify_labels = [f"{sample_to_cohort[idx]}_{int(label)}"
                           for idx, label in zip(df_train_aligned.index, train_labels)]

        try:
            train_indices, val_indices = train_test_split(
                range(len(df_train_aligned)),
                test_size=val_ratio,
                stratify=stratify_labels,
                random_state=args.seed
            )
        except ValueError as e:
            print(f"Warning: Stratification by cohort+response failed ({e})")
            print("Falling back to response-only stratification...")
            train_indices, val_indices = train_test_split(
                range(len(df_train_aligned)),
                test_size=val_ratio,
                stratify=train_labels.values,
                random_state=args.seed
            )

        df_train_split = df_train_aligned.iloc[train_indices]
        df_val_aligned = df_train_aligned.iloc[val_indices]
        train_labels_split = train_labels.iloc[train_indices]
        val_labels = train_labels.iloc[val_indices]

        val_labels_df = pd.DataFrame({
            0: (val_labels == 0).astype(int),
            1: (val_labels == 1).astype(int)
        }, index=val_labels.index)

        train_labels_split_df = pd.DataFrame({
            0: (train_labels_split == 0).astype(int),
            1: (train_labels_split == 1).astype(int)
        }, index=train_labels_split.index)

        print(f"\nTraining set: {len(df_train_split)} samples")
        print(f"  Response: {np.sum(train_labels_split == 1)}")
        print(f"  No Response: {np.sum(train_labels_split == 0)}")

        print(f"\nValidation set: {len(df_val_aligned)} samples")
        print(f"  Response: {np.sum(val_labels == 1)}")
        print(f"  No Response: {np.sum(val_labels == 0)}")
        print(f"{'='*80}\n")
    else:
        df_train_split = df_train_aligned
        train_labels_split_df = train_labels_df
        df_val_aligned = None
        val_labels_df = None

    # Fine-tune the model
    print("\n" + "="*80)
    print("FINE-TUNING COMPASS MODEL")
    print("="*80)

    ft_args = {
        'mode': args.mode,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'max_epochs': args.max_epochs,
        'patience': args.patience,
        'seed': args.seed,
        'load_decoder': False,
        'with_wandb': args.with_wandb,
        'wandb_project': args.wandb_project,
        'wandb_entity': args.wandb_entity,
        'wandb_dir': args.wandb_dir,
        'wandb_group': wandb_group,
        'wandb_name': f"{setting}_{test_group}",
        'wandb_config': {
            'setting': setting,
            'mode': args.mode,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'max_epochs': args.max_epochs,
            'patience': args.patience,
            'seed': args.seed,
            'load_decoder': False,
            'test_group': str(test_group),
            'train_groups': [str(g) for g in all_groups if g != test_group],
            'num_groups': len(all_groups),
            'val_ratio': val_ratio,
            'clinical_features_file': args.clinical_features_file,
            'treatment_gating_enabled': args.treatment_gating_enabled,
            'concept_alignment_loss_scale': args.concept_alignment_loss_scale,
            'concept_alignment_mode': args.concept_alignment_mode,
            'pathway_consistency_loss_scale': args.pathway_consistency_loss_scale,
            'auxiliary_task_loss_scale': args.auxiliary_task_loss_scale,
            'biomarker_attention_enabled': args.biomarker_attention_enabled,
        },
        'work_dir': output_dir,
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

    print("Starting fine-tuning...")
    if val_ratio == 0:
        finetuner.tune(
            dfcx_train=df_train_aligned,
            dfy_train=train_labels_df,
            min_mcc=0.8
        )
    else:
        finetuner.tune_with_test(
            dfcx_train=df_train_split,
            dfy_train=train_labels_split_df,
            dfcx_test=df_val_aligned,
            dfy_test=val_labels_df,
            min_mcc=0.8
        )

    # Evaluate on test data
    print("\n" + "="*80)
    print(f"EVALUATING ON TEST GROUP ({test_group})")
    print("="*80)

    print("Making predictions on test data...")
    _, df_pred = finetuner.predict(df_test_aligned, batch_size=128, num_workers=args.num_workers)

    if df_pred.shape[1] == 2:
        predictions = df_pred.iloc[:, 1].values
    else:
        predictions = df_pred.values.flatten()

    print("\n" + "="*80)
    print(f"COMPASS EVALUATION RESULTS ON GROUP '{test_group}'")
    print("="*80)
    print(f"Group: {test_group}")
    print(f"Samples: {len(df_test_aligned)}")
    print(f"Responders: {np.sum(test_labels == 1)}")
    print(f"Non-responders: {np.sum(test_labels == 0)}")
    print()

    auc = roc_auc_score(test_labels, predictions)
    pred_binary = (predictions > 0.5).astype(int)
    accuracy = accuracy_score(test_labels, pred_binary)
    precision = precision_score(test_labels, pred_binary, zero_division=0)
    recall = recall_score(test_labels, pred_binary, zero_division=0)
    f1 = f1_score(test_labels, pred_binary, zero_division=0)

    print("Performance Metrics:")
    print(f"  AUC:       {auc:.3f}")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-score:  {f1:.3f}")
    print()

    print("Prediction Statistics:")
    print(f"  Mean prediction score: {np.mean(predictions):.3f}")
    print(f"  Std prediction score:  {np.std(predictions):.3f}")
    print(f"  Min prediction score:  {np.min(predictions):.3f}")
    print(f"  Max prediction score:  {np.max(predictions):.3f}")
    print("="*80)

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
            'test_n_responders': int(np.sum(test_labels == 1)),
            'test_n_non_responders': int(np.sum(test_labels == 0))
        }
        finetuner.log_test_metrics(test_metrics)

    # Save detailed results — filename prefix based on setting
    results_df = pd.DataFrame({
        'sample_id': df_test_aligned.index,
        'true_label': test_labels.values,
        'predicted_score': predictions,
        'predicted_binary': pred_binary
    })
    group_short = str(test_group).replace('_', '').lower()
    results_file = os.path.join(results_dir, f'iatlas_{setting}_{group_short}_predictions.csv')
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(results_file, index=False)
    print(f"\nDetailed results saved to: {results_file}")

    if args.with_wandb:
        finetuner.finish_wandb()

    return {
        'setting': setting,
        'test_group': str(test_group),
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
parser = argparse.ArgumentParser(
    description='Fine-tune COMPASS model with leave-one-group-out cross-validation. '
                'Settings: loco (leave-one-cohort-out), locto (leave-one-cancer-type-out), '
                'loto (leave-one-ICI-target-out).')

# --- Setting ---
parser.add_argument('--setting', type=str, default='loco',
                    choices=['loco', 'loto', 'locto'],
                    help='Cross-validation setting: loco (cohort), loto (ICI target), '
                         'locto (cancer type) (default: loco)')

# --- Cohort pre-filter (applied before setting-specific group splitting) ---
parser.add_argument('--all_cohorts', type=str, default=None,
                    help='Comma-separated list of cohort names (Dataset column) to restrict '
                         'the data pool. Applied before setting-specific grouping, so LOTO '
                         'and LOCTO will only use samples from these cohorts.')

# --- Group selection ---
parser.add_argument('--groups', type=str, default=None,
                    help='Comma-separated list of group identifiers to include. '
                         'For loco: cohort names. For locto: TCGA cancer type strings. '
                         'For loto: ICI target strings (e.g. "PD1,PD-L1"). '
                         'If not specified, all groups are auto-discovered from data.')
parser.add_argument('--test_group', type=str, default=None,
                    help='Single group to test. If specified, runs only this fold. '
                         'For loto, pass as ICI target string (e.g. "PD1").')

# --- Data paths ---
parser.add_argument('--gene_exp_file', type=str,
                    default='data/gene_exp.tsv',
                    help='Path to gene expression data file')
parser.add_argument('--model_path', type=str, default='models/pretrainer.pt',
                    help='Path to pre-trained model')
parser.add_argument('--output_dir', type=str, default='models',
                    help='Directory to save fine-tuned models (default: models)')
parser.add_argument('--results_dir', type=str, default='results/finetune/default',
                    help='Directory to save prediction results (default: results/finetune/default)')

# --- Training hyperparameters ---
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

# --- Clinical feature integration parameters ---
parser.add_argument('--clinical_features_file', type=str, default=None,
                    help='Path to clinical features TSV file. Required for loto (ICI target) setting.')
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

# Validate clinical_features_file is provided for settings that need it
if args.setting == 'loto' and args.clinical_features_file is None:
    raise ValueError(f"--clinical_features_file is required for --setting {args.setting}")

# Set random seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Discover groups based on setting
print(f"Setting: {args.setting.upper()} (Leave-One-{args.setting[2:].upper()}-Out)")
print("Discovering groups from data...")

all_cohorts_filter = [c.strip() for c in args.all_cohorts.split(',')] if args.all_cohorts else None

if args.setting == 'loco':
    discovered_groups, group_assignments, df_sample_info_filtered = \
        discover_cohorts(args.clinical_features_file, all_cohorts_filter)
elif args.setting == 'locto':
    discovered_groups, group_assignments, df_sample_info_filtered = \
        discover_cancer_types(args.clinical_features_file, all_cohorts_filter)
elif args.setting == 'loto':
    discovered_groups, group_assignments, df_sample_info_filtered = \
        discover_ici_target_groups(args.clinical_features_file, all_cohorts_filter)

# group_assignments is already aligned to df_sample_info_filtered from discovery
group_assignments = group_assignments.reindex(df_sample_info_filtered.index)

# Parse and validate --groups
if args.groups is not None:
    # Parse the comma-separated string; cast to int only for loto
    raw_groups = [g.strip() for g in args.groups.split(',')]
    all_groups = raw_groups
    for g in all_groups:
        if g not in discovered_groups:
            raise ValueError(f"Group '{g}' not found in data. Available: {discovered_groups}")
else:
    # Use all auto-discovered groups
    all_groups = discovered_groups

print(f"Using {len(all_groups)} groups:")
for g in all_groups:
    n_samples = int((group_assignments == g).sum())
    print(f"  {g}: {n_samples} samples")

# Resolve test_group
if args.test_group is not None:
    test_group_val = args.test_group
    if test_group_val not in all_groups:
        raise ValueError(f"test_group '{test_group_val}' not in groups list. Available: {all_groups}")
    groups_to_test = [test_group_val]
    all_groups_pool = all_groups
else:
    groups_to_test = all_groups
    all_groups_pool = all_groups

print("\n" + "="*80)
print(f"LEAVE-ONE-{args.setting[2:].upper()}-OUT CROSS-VALIDATION")
print("="*80)
print(f"Number of folds: {len(groups_to_test)}")
print(f"Groups to test: {groups_to_test}")
print("="*80)

# Load TPM data
print("\nLoading gene expression data...")
print(f"  Gene expression file: {args.gene_exp_file}")
df_tpm_full = pd.read_csv(args.gene_exp_file, sep='\t', index_col='Run_ID')
# df_tpm_full.index = df_tpm_full.index.str.upper()
print(f"  Gene expression data shape: {df_tpm_full.shape}")

# Align TPM with filtered samples (indices normalised to uppercase)
# df_sample_info_filtered.index = df_sample_info_filtered.index.str.upper()
df_tpm_full = df_tpm_full.loc[df_sample_info_filtered.index]

# Build per-sample cancer code from TCGA_Study column
sample_to_cancer_code = df_sample_info_filtered['TCGA_Study'].map(CANCER_TYPE_TO_CODE)
missing = sample_to_cancer_code[sample_to_cancer_code.isna()].index.tolist()
if missing:
    print(f"Warning: {len(missing)} samples have no cancer code (missing TCGA_Study), defaulting to -1")
    sample_to_cancer_code = sample_to_cancer_code.fillna(-1).astype(int)

# Map Responder to binary labels
responder_map = {True: 1, False: 0}
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

expected_genes = model_template.scaler.scaler.feature_names_in_
print(f"Model expects {len(expected_genes)} genes")

# Perform cross-validation
all_results = []

import time
wandb_group = f"{args.setting}_{args.mode}_seed{args.seed}_{int(time.time())}"

for i, test_group in enumerate(groups_to_test):
    print(f"\n\n{'#'*80}")
    print(f"# FOLD {i+1}/{len(groups_to_test)}: Testing on group '{test_group}'")
    print(f"{'#'*80}\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if i > 0:
        print(f"Loading fresh model from {args.model_path}...")
        model = loadcompass(args.model_path, map_location=device)
    else:
        model = model_template

    fold_results = train_and_test(
        test_group=test_group,
        all_groups=all_groups_pool,
        group_assignments=group_assignments,
        sample_to_cancer_code=sample_to_cancer_code,
        df_tpm_full=df_tpm_full,
        df_sample_info_filtered=df_sample_info_filtered,
        labels_full=labels_full,
        model=model,
        expected_genes=expected_genes,
        args=args,
        wandb_group=wandb_group,
        val_ratio=args.val_ratio,
        setting=args.setting,
        output_dir=args.output_dir,
        results_dir=args.results_dir
    )

    all_results.append(fold_results)

# Print summary
print("\n\n" + "="*80)
print(f"LEAVE-ONE-{args.setting[2:].upper()}-OUT CROSS-VALIDATION SUMMARY")
print("="*80)
print()

summary_df = pd.DataFrame(all_results)

print("Per-Group Results:")
print("-" * 80)
print(f"{'Group':<25} {'Train N':>8} {'Test N':>8} {'AUC':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
print("-" * 80)
for _, row in summary_df.iterrows():
    print(f"{str(row['test_group']):<25} {row['n_train_samples']:>8} {row['n_test_samples']:>8} "
          f"{row['auc']:>8.3f} {row['accuracy']:>8.3f} {row['precision']:>8.3f} "
          f"{row['recall']:>8.3f} {row['f1']:>8.3f}")
print("-" * 80)

print("\nOverall Statistics:")
print(f"  Mean AUC:       {summary_df['auc'].mean():.3f} ± {summary_df['auc'].std():.3f}")
print(f"  Mean Accuracy:  {summary_df['accuracy'].mean():.3f} ± {summary_df['accuracy'].std():.3f}")
print(f"  Mean Precision: {summary_df['precision'].mean():.3f} ± {summary_df['precision'].std():.3f}")
print(f"  Mean Recall:    {summary_df['recall'].mean():.3f} ± {summary_df['recall'].std():.3f}")
print(f"  Mean F1-score:  {summary_df['f1'].mean():.3f} ± {summary_df['f1'].std():.3f}")
print()
print(f"  Total test samples: {summary_df['n_test_samples'].sum()}")
print(f"  Groups tested: {len(summary_df)}")
print("="*80)

# Save summary
summary_file = os.path.join(args.results_dir, f'iatlas_{args.setting}_summary.csv')
summary_df.to_csv(summary_file, index=False)
print(f"\nSummary results saved to: {summary_file}")

print(f"\n{args.setting.upper()} cross-validation completed successfully!")
