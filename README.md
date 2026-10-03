# MSRHGNN Prediction
## Introduction
### Author Contact Information:
Author 1: Jinzhou Wu, Email: 605318088@qq.com

Author 2: Donglin He, Email: 1364825364@qq.com

Author 3: Xin Li, Email: 2739273211@qq.com

Author 4: Rui Wang, Email: 2219312248@qq.com

Author 5:Yujuan Zhang, Email: yujuan.zhang418@gmail.com

## Data available
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



## Environment Setup
- torch 1.13.0+cu117
- numpy 1.24.4
- pandas 2.0.3
- scikit-learn 1.3.2
- networkx 2.8.4
- Python 3.8.19
- dgl 0.9.1



## Code

- data_preprocess.py: Methods of data processing

- metric.py: Metrics calculation

- pos_contrast.py: Get positive and negative samples for contrastive learning

- Contrast.py: Code of graph contrastive learning

- global_pc_grad_magnitude_positive.py: Algorithm of GradCraft

- model.py: Model of MSRHGNN

- train_DDA.py: Train the model

  

## Run the Code

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



## Acknowledgments

This project benefits from open-source implementations provided by previous studies. We sincerely appreciate the authors for sharing their code.

- https://github.com/OleCui/paper_GCGB
- https://github.com/JK-Liu7/DRMAHGC







## Before running

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

## Outputs

Training prints per-epoch metrics and final AUC summaries. Whenever a fold's AUC
improves, it writes `label<fold>.npy` and `score<fold>.npy` under
`Results/<dataset>/<timestamp>/`. Labels and scores correspond by row.
These are evaluation artifacts, not a saved model checkpoint.

The defaults run 10 folds with 300 epochs each. End-to-end runtime has not been
measured for this quick-start; it depends on the dataset and GPU.
