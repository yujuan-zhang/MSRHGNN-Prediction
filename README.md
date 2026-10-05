# MSRHGNN Prediction

## What it does

Train and evaluate a model for drug–disease associations using drug, disease
and protein information together. Known associations and similarity/feature
tables form a heterogeneous graph: drugs, diseases and proteins are different
node types, connected by their corresponding relations.

The workflow is:

```text
Benchmark tables → heterogeneous graph → structural and metapath augmentation
                 → representation learning and association classification
                 → held-out drug–disease scores and evaluation metrics
```

The augmentations provide different graph views for learning useful node
representations. The classifier then scores drug–disease pairs. This is a
benchmark for studying association prediction; the training script evaluates
cross-validation folds rather than serving an API for new cases.

Start with `check_data.py` to check the required files, matrix sizes and
association indices. Training needs the documented CUDA/DGL environment.
It saves labels and scores for evaluation, not a ready-to-load model checkpoint.
The B-dataset loader and focused sampling checks were performed; full CUDA
training was not run. The sampling implementation is described later in this README.

## Input

### Before running

Run from the repository root. The current code uses CUDA explicitly; it does not
provide a CPU fallback. Keep the Python and CUDA/DGL versions listed above.
Download the benchmark data using the Data available link. `get_data()` actually
reads the following files from `data/<dataset>/`:

```text
DrugFingerprint.csv             DrugGIP.csv
drug_ie_sim.csv                  DiseasePS.csv
DiseaseGIP.csv                   disease_ie_sim.csv
Protein_sequence.csv            ProteinGIP_Drug.csv
ProteinGIP_Disease.csv           DrugDiseaseAssociationNumber.csv
DrugProteinAssociationNumber.csv ProteinDiseaseAssociationNumber.csv
drug_f512_feature.csv            disease_f512_feature.csv
protein_f512_feature.csv
```

Similarity/sequence CSV readers drop the first column; association tables are
read as integers; the three `*_f512_feature.csv` files are read without a header.
Use the original benchmark files and their row ordering. These requirements
come from `data_preprocess.py`; the names in the earlier dataset description
are not a substitute for this list. Although `train_DDA.py` assigns
`SGMAE/Embedding/<dataset>/` to `GAE_data_dir`, the current loader reads these
feature files from `data/<dataset>/` instead.

## Output

### Outputs

Training prints per-epoch metrics and final AUC summaries. Whenever a fold's AUC
improves, it writes `label<fold>.npy` and `score<fold>.npy` under
`Results/<dataset>/<timestamp>/`. Labels and scores correspond by row.
These are evaluation artifacts, not a saved model checkpoint.

The defaults run 10 folds with 300 epochs each. End-to-end runtime has not been
measured for this quick-start; it depends on the dataset and GPU.

## Try it

### Environment Setup

- torch 1.13.0+cu117
- numpy 1.24.4
- pandas 2.0.3
- scikit-learn 1.3.2
- networkx 2.8.4
- Python 3.8.19
- dgl 0.9.1

### Run the Code

The command to train MSRHGNN on the B-dataset, C-dataset or F-dataset is as follows.

B-dataset:

```python
python train_DDA.py --dataset B-dataset
```

C-dataset:

```python
python train_DDA.py --dataset C-dataset
```

F-dataset:

```python
python train_DDA.py --dataset F-dataset
```

### Check downloaded data before training

This check needs only Python's standard library; it does not import PyTorch or
DGL, start CUDA training, or change data files:

```bash
python check_data.py --dataset B-dataset
# Check another downloaded benchmark:
python check_data.py --dataset C-dataset
```

Paths default to `data/<dataset>/` beside this script, so it can also be launched
from another directory. Use `--data-dir /path/to/B-dataset` for a separate data
folder. Success prints a JSON report and exits with code 0; missing files,
wrong matrix dimensions, nonnumeric/nonfinite cells, or invalid association
indices exit with code 2 and an explanatory error. The B-dataset has 269 drugs,
598 diseases and 1,021 proteins. The checked local association CSV has 18,410
drug–disease rows; the earlier paper-summary table reports 18,416 and is retained
as published. The checker reports the local file contents.

Despite the `f512_feature` filenames, the current loader expects square
entity-by-entity matrices: the drug, disease and protein files must be
269 × 269, 598 × 598 and 1,021 × 1,021 for B-dataset, with no header or index
column. They are not directly consumable 512-column embeddings. The other nine
similarity files have a header and a first index column, which the loader drops.
Association CSVs have a header and two zero-based integer columns;
`ProteinDiseaseAssociationNumber.csv` is read in disease, protein order.
The check validates file formats and index ranges, not biological correctness,
entity-ID alignment across modalities, negative sampling, or model accuracy.

### Introduction

### Data available

We evaluate the performance of our proposed method on three widely-used benchmarks: B-dataset, C-dataset, and F-dataset. These datasets are also available for download through Google Drive for your convenience.(https://drive.google.com/drive/folders/10bFArKqQT1DZEiugsn0RByUnkdXiPAKU?usp=sharing). After downloading, please create a data folder and place the datasets inside it. Below are the specific descriptions of the datasets.

| Dataset   | Drug | Disease | Protein | Drug-Disease | Drug-Protein | Disease-Protein | Sparsity |
| --------- | ---- | ------- | ------- | ------------ | ------------ | --------------- | -------- |
| B-dataset | 269  | 598     | 1021    | 18416        | 3110         | 5898            | 11.45    |
| C-dataset | 663  | 409     | 993     | 2532         | 3672         | 10691           | 0.93     |
| F-dataset | 592  | 313     | 2741    | 1933         | 3152         | 47470           | 1.04     |

- *DrugFingerprint.csv*: The drug fingerprint similarities between each drug pairs.
- *DrugGIP.csv*: The drug Gaussian interaction profile (GIP) kernel similarities between each drug pairs.
- *drug_ie_sim.csv*: The drug similarity between each drug pair based on information entropy.
- *DiseasePS.csv*: The disease phenotype similarities between each disease pairs.
- *DiseaseGIP.csv*: The disease GIP kernel similarities between each disease pairs.
- *disease_ie_sim.csv*: The disease similarity between each disease pair based on information entropy.
- *DrugDiseaseAssociationNumber.csv* : The known drug-disease associations.
- *DrugProteinAssociationNumber.csv* : The known drug-protein associations.
- *ProteinDiseaseAssociationNumber.csv* : The known disease-protein associations.
- *Drug_mol2vec.csv* : The 300-dimensional mol2vec embeddings of drugs
- *DiseaseFeature.csv* : The 64-dimensional MeSH embeddings of diseases.
- *Protein_ESM.csv* : The 320-dimensional ESM-2 embeddings of proteins.

The datasets used in this paper are consistent with those employed in previous studies. https://github.com/OleCui/paper_GCGB

### Code

- data_preprocess.py: Methods of data processing

- metric.py: Metrics calculation

- pos_contrast.py: Get positive and negative samples for contrastive learning

- Contrast.py: Code of graph contrastive learning

- global_pc_grad_magnitude_positive.py: Algorithm of GradCraft

- model.py: Model of MSRHGNN

- train_DDA.py: Train the model

### Acknowledgments

This project benefits from open-source implementations provided by previous studies. We sincerely appreciate the authors for sharing their code.

- https://github.com/OleCui/paper_GCGB
- https://github.com/JK-Liu7/DRMAHGC

### Focused validation

The bundled B-dataset files passed the actual `get_data()` loader in a local
CPU check. The local C/F dataset copies were incomplete (the first missing
file was `drug_ie_sim.csv`); verify the entire required file list after download.
No CUDA training was performed.

### Sampling implementation

These publication files retain the existing GitHub KMeans-based negative sampling.
The focused source repair selects disease features from `di_fusion` rather than
`dr_fusion`. The data loader and a focused negative-sampling check were verified;
full CUDA training and model accuracy were not validated.

