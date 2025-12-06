# AIRR-ML-25: Adaptive Immune Profiling Challenge - Data

## Dataset Overview

- **Total Files**: 7832 files
- **Total Size**: 19.94 GB
- **File Types**: tsv, csv
- **License**: [Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

## Download Command

```bash
kaggle competitions download -c adaptive-immune-profiling-challenge-2025
```

---

## Dataset Description

### Files

### Training and test datasets

There are a total of 8 training datasets stored under **train_datasets** with the following naming convention: **train_dataset_1**, **train_dataset_2** and so on. For each training dataset, one or more corresponding test dataset(s) are stored under **test_datasets** with a similar naming convention pattern: **test_dataset_1**, **test_dataset_2** and so on. When multiple test datasets are provided per training dataset, they follow a naming convention like **test_dataset_7_1**, **test_dataset_7_2** and so on.

### Files of examples and metadata

Each training dataset contains a metadata file (**metadata.csv**) that lists the filenames of examples, their unique identifiers, and a target label for each example. The metadata file has a comma-separated value (CSV) format with the following fields: **repertoire_id**, **filename**, **label_positive**. The **label_positive** is the target label that indicates whether the example is positive-labeled (True for positive-labelled and False for negative-labelled). There may be additional metadata in the metadata files in some cases that the participants may want to use. All the fields of metadata files are explained below.

---

## Metadata Fields

### Core metadata fields (always provided)

- `repertoire_id`: A unique identifier for each repertoire/example
- `filename`: Filename of each repertoire
- `label_positive`: Target label to be used for training the models and for prediction. `True` represents the target variable of interest.

### Dataset-specific metadata fields (sometimes provided)

- `study_group_description`: Status indicating whether an example is diabetic, or first degree relative of diabetic (FDR), second degree relative of a diabetic (SDR), or non-diabetic controls (CTRL)
- `sex`: biological sex at birth
- `age`: age in number of years
- `race`: Racial group of example (as defined by NIH)
- `sequencing_run_id`: ID of sequencing run assigned by the sequencing facility.
- `A`, `B`, `C`: Precise genetic variation for HLA class-I genes
- `DPA1`, `DPB1`, `DQA1`, `DRB1`, `DRB3`, `DRB4`, `DRB5`: Precise genetic variation for HLA class-II genes

---

## Potential roles of different metadata

### Age, sex, and race
These factors are often associated with certain health conditions and can introduce bias into models. For example, some diseases are more common in older people or in a particular sex. While this information can be useful, one should be mindful of how one uses it to avoid building a model that simply learns these associations instead of the true biological patterns.

### HLA genes (as indicated above)
HLA stands for Human Leukocyte Antigen. It is a key part of immune system and specific HLA genes can directly influence susceptibility to certain diseases. Unlike age or sex which may be indirect indicators, HLA type can be a more direct causal factor for some diseases. Cleverly using this information could be a powerful way to improve your predictions.

### Sequencing runs (as indicated by sequencing_run_id above)
Systematic differences can creep in during data collection or processing between different sequencing runs. It is crucial to ensure your model is learning real biological patterns, not such technical artefacts.

---

## The file format of examples

The examples (each file in each training/test dataset) have a tab-separated value (tsv) format. The example tsv files contain three fields: **junction_aa**, **v_call**, **j_call**. The **junction_aa** represents the amino acid sequence of TCR beta CDR3 (IMGT junction specifically), whereas the **v_call** and **j_call** represent the V and J genes involved in producing this specific sequence. For some datasets, where available, "d_call" represents D genes and "templates" indicating the number of duplicate observations for the query sequence (`duplicate_count` in AIRR format) are also provided.

### Explanation of the fields for participants from outside the immunology domain:

#### `junction_aa`
This represents the protein "code" of a very specific segment of an immune receptor. Imagine it as a unique "barcode", which is crucial for recognising and neutralising invaders like viruses or bacteria. This segment is incredibly diverse and is what allows your immune system to fight so many different threats!

#### `v_call`, `j_call`, `d_call`
To create the incredible diversity of immune receptors, your body uses a clever genetic "shuffling" process. These fields tell us which specific "gene building blocks" (called V, D and J genes) were used to create that particular junction_aa sequence. Think of them as LEGO pieces from a large LEGO set that were combined to make that unique immune receptor segment and provide stability to junction_aa (barcode) to function properly.

---

## What am I predicting?

For each repertoire_id across all test datasets, the participants have to return a probability for the repertoire being label-positive. In addition, a ranked list of the top 50,000 unique rows (including junction_aa, v_call, and j_call) that best contribute to the optimal classification for each training dataset has to be returned, regardless of the data encoding used. Note that these label-associated sequences have to be sorted based on some form of importance scores from most important to least important; we may use only the top-n sequences from the ordered list of 50k sequences for evaluation.

---

## What to report?

See what should be reported in the submission file under "**Submission File**" in the **Overview** page. A file named `submissions.csv` should be returned. An example of such a file, `sample_submissions.csv` is provided here. Missing values in the `sample_submissions.csv` are represented by a special float '-999.0' as Kaggle's validation does not allow missing values such as 'NaN' in submission files. As explained under "**Submission File**" in the **Overview** page, for the first 4213 rows, the `label_positive_probability` should be predicted and filled while ignoring the other fields with missing values. For the remaining rows, `junction_aa`, `v_call` and `j_call` fields should be filled while ignoring the other fields with missing values. Note that in the `sample_submissions.csv`, we filled these fields in relevant rows (4214-404213) with toy sequences and gene information. In the `submissions.csv`, the participants should replace these toy sequences with rank-ordered label-associated sequences from respective training datasets.

---

## Data Explorer Structure

```
adaptive-immune-profiling-challenge-2025/
├── test_datasets/
│   ├── test_dataset_1/
│   ├── test_dataset_2/
│   ├── ...
│   ├── test_dataset_7_1/
│   ├── test_dataset_7_2/
│   └── ...
├── train_datasets/
│   ├── train_dataset_1/
│   │   ├── metadata.csv
│   │   └── [repertoire files].tsv
│   ├── train_dataset_2/
│   │   ├── metadata.csv
│   │   └── [repertoire files].tsv
│   └── ... (8 training datasets total)
└── sample_submissions.csv (34.25 MB)
```

---

## Key Numbers

- **Total repertoires in test datasets**: 4,213
- **Total submission rows required**: 404,213
  - 4,213 rows for repertoire predictions (label_positive_probability)
  - 400,000 rows for rank-ordered sequences (50,000 per each of 8 training datasets)
- **Training datasets**: 8
- **Columns in data**: 29.1k columns (across all files)
