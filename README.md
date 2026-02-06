# BioCOMPASS: Integrating Biomarkers into Transformer-Based Immunotherapy Response Prediction

**BioCOMPASS** extends the COMPASS framework by integrating clinical biomarkers and treatment information to enhance immunotherapy response prediction. This repository builds upon [COMPASS](https://github.com/mims-harvard/COMPASS) by adding support for:

- 🔬 **Treatment indicators** (PD-1, CTLA-4, combination therapy)
- 🧬 **Cell type biomarkers** (62 features from immune cell signatures)
- 🔀 **Pathway activity scores** (42 features from CTLA4/PD1 pathways)
- 📊 **Auxiliary biomarkers** (TIDE, IPRES, phenotype scores)

## Key Features

### 🌟 Clinical Feature Integration

BioCOMPASS incorporates five strategies to integrate clinical knowledge:

1. **Treatment-Aware Gating**: Modulates biological concepts based on treatment type
2. **Concept-Biomarker Alignment**: Aligns learned concepts with external biomarkers
3. **Pathway Consistency**: Ensures pathway predictions match known biology
4. **Auxiliary Task Learning**: Multi-task learning on TIDE, IPRES, phenotype
5. **Biomarker-Guided Attention**: Cross-attention between biomarkers and gene encodings

### 📦 What's Included

- ✅ **Clinical data loader** - Handles treatment, biomarker, and pathway features
- ✅ **Integration modules** - TreatmentGating, BiomarkerAttention, and more
- ✅ **Production scripts** - Ready-to-use training and evaluation pipelines
- ✅ **Example notebooks** - Demonstrations with real data

---

## Installation

### Create BioCOMPASS Conda Environment

BioCOMPASS uses Python 3.10. Follow these steps to set up your environment:

```bash
# Clone the repository
git clone https://github.com/hashimsayed0/BioCOMPASS.git
cd BioCOMPASS

# Create a new conda environment with Python 3.10
conda create -n biocompass python=3.10 -y

# Activate the environment
conda activate biocompass

# Install dependencies
pip install -r requirements.txt
```

---

## Data Acquisition

BioCOMPASS uses immunotherapy clinical trial data from the **CRI iAtlas** portal hosted on Synapse.

### Downloading Data from Synapse

1. **Create a Synapse account** at [https://www.synapse.org](https://www.synapse.org) if you don't have one
2. **Access the CRI iAtlas data repository**: [syn24200710](https://www.synapse.org/Synapse:syn24200710)
3. **Make sure you're in the right data folder**: `Files/Molecular Response to Immune Checkpoint Inhibitors`
4. **Download the following three files**:

   | File Name | Description | Required For |
   |-----------|-------------|--------------|
   | `iatlas-ici-sample_info.tsv` | Sample metadata and response labels | Labels file |
   | `iatlas-ici-features.tsv` | Clinical features and biomarkers | Clinical features |
   | `iatlas-ici-hgnc_tpm.tsv` | Gene expression (TPM-normalized) | Gene expression data |

5. **Place the files in the `data/` directory** of your BioCOMPASS installation:
   ```bash
   mkdir -p data
   # Move or copy downloaded files to data/
   mv iatlas-ici-sample_info.tsv data/labels.tsv
   mv iatlas-ici-features.tsv data/clinical_features.tsv
   mv iatlas-ici-hgnc_tpm.tsv data/tpm.tsv
   ```

**Note**: All three required files are located in the `Files/Molecular Response to Immune Checkpoint Inhibitors` folder within the Synapse repository.

---

## Data Preparation

BioCOMPASS requires three main input files in the `data/` directory:

### 1. Gene Expression Data (`gene_exp.tsv`)
- TPM-normalized gene expression matrix
- Format: First column = cancer type code, remaining 15,672 columns = genes
- Each row represents one patient sample

### 2. Labels File (`labels.tsv`)
- Sample metadata and response labels
- Required columns: `Run_ID`, `Dataset` (cohort), `Responder` (True/False), `TCGA_Study` (cancer type)

### 3. Clinical Features (`clinical_features.tsv`)

The clinical features file should contain **139 total features** across 4 categories:

#### Treatment Indicators (4 features)
- `aPD1_Tx` - Anti-PD1 treatment
- `aCTLA4_Tx` - Anti-CTLA4 treatment
- `aCTLA4_aPD1_Tx` - Combination therapy
- `Prior_aCTLA4_Tx` - Prior CTLA4 treatment

#### Cell Type Biomarkers (62 features)
B cell markers, T cell markers, Myeloid markers, NK cells:
- `IglesiaVincent_BCell`, `Palmer_BCell`, `Bindea_BCells`, `Schmidt_BCell`
- `GO_BCR_Signaling`, `Fan_IGG`, `Rody_TNBC_BCell`, `Vincent_Plasma_Cells`
- `Palmer_CD8`, `IglesiaVincent_CD8`, `Bindea_CD8_TCells`, `IglesiaVincent_TCell`
- `Palmer_TCell`, `Bindea_TCells`, `GO_TCR_Signaling`, `Rody_TNBC_TCell`
- `Bindea_THelper`, `Bindea_Th1_Cells`, `Bindea_Th2_Cells`, `Bindea_Th17_Cells`
- `Bindea_TReg`, `Bindea_Cytotoxic_Cells`, `Bindea_Tcm`, `Bindea_Tem`
- `Bindea_TFH`, `Bindea_Tgd`, `Rody_LCK`, `IglesiaVincent_MacTh1`, `TIDE_CD8`
- `IglesiaVincent_CD68`, `Beck_Mac_CSF1`, `Bindea_Macrophages`, `Bindea_aDC`
- `Bindea_DC`, `Bindea_iDC`, `Bindea_Neutrophils`, `Bindea_pDC`, `CSF1_Response`
- `Bindea_Eosinophils`, `Bindea_Mast_Cells`, `Bindea_NK_CD56bright`
- `Bindea_NK_CD56dim`, `Bindea_NK_Cells`

#### Pathway Activity Scores (42 features)
CTLA4 pathway scores (28) + PD1 pathway scores (14):
- `BIOCARTA_CTLA4_V_*` (14 features for different T cell signatures)
- `REACTOME_CTLA4_V_*` (14 features for different T cell signatures)
- `REACTOME_PD1_V_*` (14 features for different T cell signatures)

#### Auxiliary Biomarkers (31 features)
TIDE scores, IPRES signatures, phenotype markers:
- **TIDE (10)**: `TIDE`, `TIDE_IFNG`, `TIDE_MSI`, `TIDE_CD274`, `TIDE_Dysfunction`, `TIDE_Exclusion`, `TIDE_MDSC`, `TIDE_CAF`, `TIDE_TAM_M2`, `TIDE_CTL`
- **IPRES (4)**: `Hugo_IPRES26`, `Hugo_IPRES22`, `Hugo_IPRES08`, `Hugo_IPRES06`
- **Phenotype (16)**: `Rody_IL8`, `ICR`, `IE_Specific`, `ID_Specific`, `Miracle`, `KardosChai_ImSuppress`, `Prat_Claudin`, `KardosChai_EMT_DOWN`, `KardosChai_EMT_UP`, `Chan_TIC`, `LIexpression_Score`, `TGFB_Score`, `Module3_IFN_Score`, `Chang_Serum_Response_Up`, `Cytolytic_Score`, `IMPRES`

**Note**: See `compass/dataloader/clinical_data.py` for the complete feature list and `data/clinical_features.tsv` for an example file.

---

## Quick Start

### Leave-One-Cohort-Out Cross-Validation

Use the provided script (or configuration in launch.json) for comprehensive evaluation across cohorts:

```bash
python tune_test_loco.py \
    --gene_exp_file data/tpm.tsv \
    --labels_file data/labels.tsv \
    --model_path model/pretrainer.pt \
    --clinical_features_file data/clinical_features.tsv \
    --cohorts "Gide_Cell_2019,HugoLo_IPRES_2016,IMvigor210,IMmotion150,Kim_NatMed_2018,Liu_NatMed_2019,Riaz_Nivolumab_2017,VanAllen_antiCTLA4_2015" \
    --test_cohort "Gide_Cell_2019" \
    --treatment_gating_enabled True \
    --concept_alignment_loss_scale 0.1 \
    --pathway_consistency_loss_scale 0.05 \
    --auxiliary_task_loss_scale 0.1 \
    --biomarker_attention_enabled False \
    --batch_size 16 \
    --max_epochs 25 \
    --with_wandb True
```

**Key Clinical Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--clinical_features_file` | Path to clinical features TSV | None |
| `--treatment_gating_enabled` | Enable treatment-aware gating | False |
| `--treatment_gating_hidden_dim` | Hidden dimension for gating network | 32 |
| `--concept_alignment_loss_scale` | Concept-biomarker alignment weight | 0.0 |
| `--concept_alignment_mode` | Alignment mode (manual/correlation/learnable) | manual |
| `--pathway_consistency_loss_scale` | Pathway consistency loss weight | 0.0 |
| `--auxiliary_task_loss_scale` | Auxiliary task loss weight | 0.0 |
| `--biomarker_attention_enabled` | Enable biomarker-guided attention | False |
| `--biomarker_attention_dim` | Attention dimension | 32 |
| `--biomarker_attention_heads` | Number of attention heads | 4 |

---

## Clinical Features File Format

The clinical features TSV file should contain the following columns:

```
sample_name    aPD1_Tx    aCTLA4_Tx    aCTLA4_aPD1_Tx    Prior_aCTLA4_Tx    [cell_type_biomarkers...]    [pathway_scores...]    [auxiliary_biomarkers...]
SAMPLE_001   1          0            0                  0                   0.523                        0.234                   0.456
SAMPLE_002   0          1            0                  0                   0.612                        0.345                   0.567
```

**Feature Categories:**
- **Treatment indicators** (4): Binary flags for treatment type
- **Cell type biomarkers** (62): Immune cell signature scores
- **Pathway scores** (42): CTLA4/PD1 pathway activity
- **Auxiliary biomarkers** (31): TIDE, IPRES, phenotype scores

See `data/clinical_features.tsv` for a complete example.

---

## Clinical Feature Integration Strategies

### Strategy 1: Treatment-Aware Gating
Modulates the 44 biological concepts based on treatment type (PD-1, CTLA-4, combo).

```python
ft_args = {
    'treatment_gating_enabled': True,
    'treatment_gating_hidden_dim': 32,
}
```

### Strategy 2: Concept-Biomarker Alignment
Aligns COMPASS concepts with external biomarkers using semantic mappings.

```python
ft_args = {
    'concept_alignment_loss_scale': 0.1,
    'concept_alignment_mode': 'manual',  # or 'correlation', 'learnable'
}
```

### Strategy 3: Pathway Consistency
Ensures predicted pathway activities match known biological pathways.

```python
ft_args = {
    'pathway_consistency_loss_scale': 0.05,
}
```

### Strategy 4: Auxiliary Task Learning
Multi-task learning on TIDE, IPRES, and phenotype predictions.

```python
ft_args = {
    'auxiliary_task_loss_scale': 0.1,
}
```

### Strategy 5: Biomarker-Guided Attention
Cross-attention mechanism between biomarkers and gene encodings.

```python
ft_args = {
    'biomarker_attention_enabled': True,
    'biomarker_attention_dim': 32,
    'biomarker_attention_heads': 4,
}
```

---

## Example Workflows

### Training with All Clinical Strategies

```python
from compass import FineTuner, loadcompass

model = loadcompass('./model/pretrainer.pt')

ft_args = {
    'mode': 'PFT',
    'lr': 1e-3,
    'batch_size': 16,
    'max_epochs': 100,

    # Enable all clinical strategies
    'clinical_features_file': './data/clinical_features.tsv',
    'treatment_gating_enabled': True,
    'treatment_gating_hidden_dim': 32,
    'concept_alignment_loss_scale': 0.1,
    'concept_alignment_mode': 'manual',
    'pathway_consistency_loss_scale': 0.05,
    'auxiliary_task_loss_scale': 0.1,
    'biomarker_attention_enabled': True,
    'biomarker_attention_dim': 32,
    'biomarker_attention_heads': 4,
}

finetuner = FineTuner(model, **ft_args)
finetuner.tune(df_tpm, dfy)
finetuner.save('./model/biocompass_full.pt')
```

### Training with Selected Strategies

```python
# Example: Only treatment gating + biomarker attention
ft_args = {
    'mode': 'PFT',
    'lr': 1e-3,
    'batch_size': 16,
    'max_epochs': 100,

    'clinical_features_file': './data/clinical_features.tsv',
    'treatment_gating_enabled': True,
    'biomarker_attention_enabled': True,
}

finetuner = FineTuner(model, **ft_args)
finetuner.tune(df_tpm, dfy)
```

---

## Architecture

BioCOMPASS extends COMPASS with:

```
COMPASS Architecture
├── Input Encoder (Transformer)
├── Projector (Gene → Concepts)
├── Task Decoder (Concepts → Response)
└── NEW: Clinical Integration Modules
    ├── TreatmentGating
    ├── BiomarkerEncoder
    ├── BiomarkerGuidedAttention
    ├── PathwayPredictorHead
    ├── AuxiliaryDecoderHead
    └── ConceptAligner
```

---

## Citing Our Work

If you use BioCOMPASS, please cite both the original COMPASS paper and this extension:

**Original COMPASS:**
```
Wanxiang Shen, Thinh H. Nguyen, Michelle M. Li, Yepeng Huang, Intae Moon,
Nitya Nair, Daniel Marbach‡, and Marinka Zitnik‡.
"Generalizable AI predicts immunotherapy outcomes across cancers and treatments."
medRxiv (2025). https://www.medrxiv.org/content/10.1101/2025.05.01.25326820
```

**BioCOMPASS extension:**
```

```

---

## License

This project is licensed under the same terms as the original COMPASS project.

## Support

For questions or issues:
- Open an issue on GitHub

---

**Built on COMPASS** | Extending transformer-based immunotherapy prediction with clinical knowledge integration
