# KMDRP

`KMDRP` (Knowledge-driven Multimodal Drug Response Predictor) is a knowledge distillation-based multimodal fusion interpretable deep neural network framework. It integrates three omics modalities (RNA expression, SGA genomic alterations, and protein expression) with drug molecular information to predict tumor drug response (IC50 values). The framework addresses real-world clinical data incompleteness through a "teacher-student" architecture with task-aware variational generative modeling, enabling accurate predictions using only RNA sequencing data.

!model.jpeg "Figure 1: Overview of the KMDRP framework"

## Features

- **High Performance**: The student model (KMDRP-S) achieves state-of-the-art performance, with RMSE = 1.1930 and R² = 0.5960 on independent test sets, outperforming 8 baseline methods.
- **Clinical Applicability**: Specifically designed for real-world scenarios with incomplete omics data, enabling robust predictions using only RNA expression data.
- **Interpretability**: Utilizes integrated gradients and attention mechanisms to provide biological insights into model decisions.
- **Knowledge Distillation Architecture**: Combines a teacher model trained on complete multi-omics data with a student model using knowledge distillation.
- **Task-aware Generation**: Variational autoencoder generates missing SGA and protein features from RNA with prior constraints.
- **Clinical Validation**: Demonstrates prognostic value in clinical cohorts with survival stratification and treatment response correlation.

## Getting Started

### Prerequisites
- Python >= 3.8.x
- PyTorch >= 1.10.0
- RDKit >= 2022.09.5 (for molecular fingerprint generation)
- Transformers >= 4.30.0 (for ChemBERTa)
- Other dependencies as listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/bowei-color/KMDRP.git
cd KMDRP
```

2. Create and activate a conda environment:
```bash
conda create -n kmdrp python=3.8
conda activate kmdrp
```

3. Install PyTorch (adjust for your CUDA version):
```bash
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0
```

4. Install other dependencies:
```bash
pip install -r requirements.txt
```

5. Install RDKit for molecular processing:
```bash
conda install -c conda-forge rdkit
```

### Usage

#### Training the Teacher Model (with complete multi-omics data):
```bash
python train_teacher.py \
    --rna_path data/rna_expression.csv \
    --sga_path data/sga_alterations.csv \
    --protein_path data/protein_expression.csv \
    --drug_smiles_path data/drug_smiles.csv \
    --output_dir models/teacher/ \
    --epochs 100 \
    --batch_size 32
```

#### Training the Student Model (with RNA-only data):
```bash
python train_student.py \
    --rna_path data/rna_expression.csv \
    --drug_smiles_path data/drug_smiles.csv \
    --teacher_model_path models/teacher/best_model.pth \
    --output_dir models/student/ \
    --epochs 100 \
    --batch_size 32
```

#### Making Predictions:
```bash
python predict.py \
    --model_path models/student/best_model.pth \
    --rna_path new_rna_data.csv \
    --drug_smiles_path new_drugs.csv \
    --output predictions.csv
```

## Input Structure

### Cell Line Data Format (CSV)
Three separate files for the three omics modalities:

**RNA Expression (rna_expression.csv):**
```csv
cell_line_id,ic50_value,ENSG000001,ENSG000002,ENSG000003,...
A549,2.34,10.235,1.378,3.546,...
MCF7,1.89,8.731,0.556,6.034,...
```

**SGA Alterations (sga_alterations.csv):**
```csv
cell_line_id,ic50_value,TP53,EGFR,KRAS,BRCA1,...
A549,2.34,1,0,1,0,...
MCF7,1.89,0,1,0,1,...
```

**Protein Expression (protein_expression.csv):**
```csv
cell_line_id,ic50_value,AKT1,MAPK1,MTOR,STAT3,...
A549,2.34,5.67,3.21,4.56,2.89,...
MCF7,1.89,4.32,5.67,3.45,4.12,...
```

### Drug Data Format (drug_smiles.csv)
```csv
drug_id,smiles,morgan_fp,properties
Drug001,CC(=O)OC1=CC=CC=C1C(=O)O,010101010...,180.16,1.21,4,1
Drug002,CN1C=NC2=C1C(=O)N(C(=O)N2C)C,101010101...,194.19,0.77,5,2
```

**Note**: The morgan_fp column should contain 2048-bit Morgan fingerprints, and properties should include molecular weight, logP, hydrogen bond acceptors, hydrogen bond donors, etc.

## Documentation

### Model Architecture

**Teacher Model (KMDRP-T):**
1. **Multi-omics Fusion Module**: Hierarchical multi-head attention network integrating:
   - RNA expression profiles
   - SGA (somatically altered gene) profiles including mutations and copy number variations
   - Protein expression data
2. **Drug Representation Module**:
   - Structure-property fusion network for molecular fingerprints and physicochemical properties
   - Transformer-based SMILES semantic modeling using ChemBERTa
3. **Fusion Regression Module**: MLP for IC50 value prediction

**Student Model (KMDRP-S):**
1. **Task-aware Variational Generation**: VAE generating missing SGA and protein features from RNA
2. **Knowledge Distillation**: Inherits frozen modules from teacher model
3. **Dual Supervision**: Optimized with both hard labels (real IC50) and soft labels (teacher predictions)

### Key Components

- **Attention Mechanisms**: Multi-head attention for cross-omics and cross-modal fusion
- **Prior-constrained VAE**: Generative modeling with biological prior knowledge
- **Integrated Gradients**: Feature importance attribution for interpretability
- **ChemBERTa Integration**: Pre-trained chemical language model for drug semantics

## Results

### Model Performance Comparison
| Model | RMSE | R² | Pearson | Spearman |
|-------|------|----|---------|----------|
| KMDRP-S (Student) | **1.1930** | **0.5960** | **0.7861** | **0.8498** |
| KMDRP-T (Teacher) | 1.2052 | 0.5877 | 0.7811 | 0.8455 |
| NeRD | 1.2300 | 0.5668 | 0.7582 | 0.8329 |
| BANDRP | 1.2507 | 0.5617 | 0.7775 | 0.8439 |
| DeepCDR | 1.2482 | 0.5697 | 0.7635 | 0.8369 |

### Clinical Validation
- **Survival Stratification**: Significant separation in overall survival (OS: p<0.05) and progression-free survival (PFS: p<0.01)
- **Treatment Response**: Predictions correlate with clinical response assessments (p<0.05)
- **Biological Consistency**: Aligns with known molecular drivers (IDH1 mutation: better prognosis, EGFR alteration: worse prognosis)
- **Personalized Therapy**: Identifies alternative treatments for IDH1 wild-type glioma patients predicted to be TMZ-resistant

### Ablation Studies
1. **Omics Modality Ablation**: 
   - RNA only: R² = 0.5530
   - RNA + generated SGA: R² = 0.5851
   - RNA + generated Protein: R² = 0.5811
   - RNA + generated SGA + generated Protein: R² = **0.5960**

2. **Knowledge Distillation Effectiveness**:
   - Student outperforms teacher in all metrics
   - Particularly effective for "hard samples" and small sample scenarios

## Datasets

The model was trained and evaluated on:
- **GDSC** (Genomics of Drug Sensitivity in Cancer): Primary training data
- **CCLE** (Cancer Cell Line Encyclopedia): Cross-validation
- **TCGA Clinical Cohort**: Independent clinical validation

Preprocessed datasets are available at: [Dataset Repository Link]

## Contributing

We welcome contributions to KMDRP! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Citation

If you use KMDRP in your research, please cite:

```bibtex
@article{yan2024kmdrp,
  title={KMDRP: Knowledge-driven Multimodal Drug Response Predictor with Clinical Applicability},
  author={Yan, Bowei and [Co-Authors]},
  journal={[Journal Name]},
  volume={[Volume]},
  number={[Number]},
  pages={[Pages]},
  year={2024},
  publisher={[Publisher]}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Dr. Bowei Yan**: boweiyan2020@gmail.com
- **GitHub Issues**: [https://github.com/bowei-color/KMDRP/issues](https://github.com/bowei-color/KMDRP/issues)
- **Project Repository**: [https://github.com/bowei-color/KMDRP](https://github.com/bowei-color/KMDRP)

## Acknowledgments

- This work was supported by [Funding Sources]
- We acknowledge the GDSC, CCLE, and TCGA consortia for providing valuable datasets
- Special thanks to all contributors and collaborators

---

*For detailed experimental results, parameter configurations, and extended analyses, please refer to the full paper.*
