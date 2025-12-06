# AIRR-ML-25: Adaptive Immune Profiling Challenge - Overview

## Competition Basic Info

- **Competition Title**: AIRR-ML-25: Adaptive Immune Profiling Challenge
- **Competition Type**: Community Prediction Competition
- **Status**: 13 days to go (as of crawl date)
- **Competition Host**: Chakravarthi Kanduri
- **URL**: https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025/

## Competition Subtitle

> Predict labels (e.g. disease, healthy) from sets of immune receptor sequences, and identify the sequences that explain the labels.

---

## Overview

In this competition, you'll develop machine learning models to simultaneously perform two tasks:

**(a)** predict the immune state (e.g. disease, healthy) of individuals based on so-called adaptive immune repertoires (sets of immune receptor sequences), and

**(b)** identify immune state-associated receptor sequences (those that explain immune state in the first task). The goal is to expedite ML-based solutions for immunodiagnostics and therapeutics discovery.

---

## Description

Imagine your body's immune system as a vast, personal army, constantly on guard against invaders like viruses and bacteria. Each soldier in this army is an "immune receptor," a tiny protein designed to recognise and fight off threats. When a new enemy (what researchers call an "antigen," like a specific virus variant) attacks, only a tiny handful out of billions of immune receptors are the perfect match to bind to it and neutralise the threat. It is like finding a needle in a haystack, but your body does it all the time. What is truly incredible is the sheer variety of these soldiers: each person has billions of unique immune receptors, each one a potential weapon against a new disease. Despite the diversity, individuals exposed to the same disease may share identical or similar immune receptors, where 'similar' can be anything from a near-perfect match to a shared structural feature or even a similar function.

Now, here is the exciting challenge: We have collections of immune receptors (called "repertoires") from many different people, and we also know if those individuals have a certain immune state (e.g. diseased or healthy).

The **big questions** for this competition:

- Can we predict a person's disease just by looking at their immune receptor sequence collections? Without knowing which receptors fight which diseases, can your machine learning models learn to identify patterns in these immune receptor collections that tell us if someone is sick or healthy?
- Can we identify the "contributing" immune receptors? If our models can predict disease, can they also tell us which specific immune receptors are most strongly linked to a particular disease? This would be like finding the star soldiers in the immune army!

Solving these problems is a huge step forward for medicine. It could lead to new ways to diagnose diseases earlier and even develop targeted treatments based on our own immune system's unique capabilities.

---

## Evaluation

For each `repertoire_id` across all test datasets, the participants has to return a probability for the repertoire being label-positive. In addition, a ranked list of the top 50,000 unique rows (including junction_aa, v_call, and j_call) that best contribute to the optimal classification for each training dataset has to be returned, regardless of the data encoding used. Note that these label-associated sequences have to be sorted based on some form of importance scores from most important to less important; we may use only top-n sequences from the ordered list of 50k sequences for evaluation. These will be used to compute the performance metrics:

- **[area under the ROC curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic)**
- **[Jaccard similarity](https://en.wikipedia.org/wiki/Jaccard_index)**

respectively, for each of the datasets. A **weighted average** of both measures across all the included datasets will be used as the basis for ranking on the leaderboard for the competition.

### Submission File

There are a total of 4213 repertoires across all test datasets. The submission file should contain a total of **404213** rows (4213 repertoire ids across test datasets and their predicted probabilities plus 50,000 rank-ordered rows per each of the training datasets with junction_aa, v_call, and j_call).

The submission file should look like this (row-index is not needed, shown only for information):

**Note:** A sample file of submissions (sample_submissions.csv) is provided under `Data` tab. As you can notice from the above image and the sample submissions file, where relevant, missing values in the submission file are to be represented by a special float '-999.0'. This is strictly needed as Kaggle's validation does not allow missing values such as 'NaN' in submission files.

Given that the prediction probabilities across all the test datasets are in `df1` (4213 rows), and the list of all rank-ordered label-associated sequences are in `df2` (400,000 rows), we assume you would combine both in a fashion similar to:

```python
combined_df = pd.concat([df1, df2], axis=0, ignore_index=True)
```

---

## Timeline

- **November 05, 2025** - Start Date. (opens at 08:00 AM CET)
- **December 17, 2025** - Final Submission Deadline (closes at 07:59 AM CET).

All deadlines are at 11:59 PM CET on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

---

## Prizes

### Monetary rewards

- **1st Place** - $ 5,000
- **2nd Place** - $ 3,000
- **3rd Place** - $ 2,000

To win the prize money, a prerequisite is that the participants make their **code open-source**.

Competition prizes are kindly sponsored by The Research Council of Norway.

### Scientific manuscript authorship

Top 10 performing participants on the final Leaderboard rankings will be invited to contribute their model descriptions, related discussions, and code to a scientific paper summarizing the competition's scientific outcome. This scientific paper has been "accepted in principle" to be published at *Nature Methods*.

---

## Code Requirements

As described above, to win the prize money, a prerequisite is that the code has to be made open-source. In addition, the top 10 submissions/teams will be invited to become co-authors in a scientific paper that involves further stress-testing of their models in a subsequent phase with many other datasets outside Kaggle platform. To enable such further analyses and re-use of the models by the community, **we strongly encourage** the participants to adhere to a [code template](https://github.com/uio-bmi/predict-airr) that we provide that enables a uniform interface of running models: https://github.com/uio-bmi/predict-airr

Ideally, all the methods can be run in a unified way, e.g.,

```bash
python3 -m submission.main --train_dir /path/to/train_dir --test_dir /path/to/test_dir --out_dir /path/to/output_dir --n_jobs 4 --device cpu
```

This requires that participants/teams adhere to the code template in `ImmuneStatePredictor` class provided in `predictor.py` by filling in their implementations within the placeholders and replacing any example code lines with actual code that makes sense.

It will also be important for the participants/teams to provide the exact requirements/dependencies to be able to containerize and run their code. If the participants/teams fork the provided repository and make their changes, it has to be remembered to also replace the dependencies in `requirements.txt` with their dependencies and exact versions.

Those participants that make use of Kaggle resources and Kaggle notebooks to make submissions are also strongly encouraged to copy the code template, particularly the `ImmuneStatePredictor` class and any utility functions from the provided code template repository and adhere to the code template to enable unified way of running different methods. Note that we provided one [public Kaggle notebook with the code template](https://www.kaggle.com/code/ckanduri/code-template).

---

## Additional resources

### Frozen research plan of this challenge

[A pre-registered protocol](https://github.com/uio-bmi/adaptive_immune_profiling_challenge_2025/blob/main/registered_report.pdf) describing all the details of the competition including extensive background information, dataset descriptions, evaluation process, and pilot data providing reference benchmarks

### What's the state-of-the-art in mining Adaptive Immune Repertoires?

- [A summary from domain experts](https://www.sciencedirect.com/science/article/pii/S2452310020300524)
- [A perspective from domain experts](https://www.sciencedirect.com/science/article/pii/S2405471224003429)

### Examples of state-of-the-art methods

- [Modern Hopfield Networks and Attention for Immune Repertoire Classification](https://doi.org/10.1101/2020.04.12.038158)
- [Immunosequencing of the T-Cell Receptor Repertoire Reveals Signatures Specific for Identification and Characterization of Early Lyme Disease](https://www.medrxiv.org/content/10.1101/2021.07.30.21261353v2.full)
- [Disease diagnostics using machine learning of B cell and T cell receptor sequences](https://pmc.ncbi.nlm.nih.gov/articles/PMC12061481/)
- [A platform for ML on adaptive immune repertoires with a wide collection of encodings and ML methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC10312379/)

---

## Acknowledgements

Adaptive Biotechnologies has generously provided ~ 500 unpublished TCRβ repertoires from a cohort of donors with known status with respect to HSV-2 infection.

Parse Biosciences has generously provided unpublished experimental antigen-specific TCR sequences for use in synthetic datasets. TCR Sequencing of 1 Million Antigen-Reactive Human T Cells in a Single Experiment, https://www.parsebiosciences.com/datasets/tcr-sequencing-of-1-million-antigen-reactive-human-t-cells-in-a-single-experiment/; Parse Biosciences, Seattle, USA, Accessed 13 March 2025.

Thanks to the AIRR-community for the shared vision and collective perspective in organizing this challenge.

---

## Citation

AIRR-ML-2025 Organizers. AIRR-ML-2025: Adaptive Immune Profiling Challenge. https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025, 2025. Kaggle.. AIRR-ML-25: Adaptive Immune Profiling Challenge. https://kaggle.com/competitions/adaptive-immune-profiling-challenge-2025, 2025. Kaggle.

---

## Participation Statistics

- **Prize Pool**: $10,000
- **Does not award Points or Medals**
- **923 Entrants**
- **178 Participants**
- **144 Teams**
- **1,607 Submissions**

---

## Tags

- Custom Metric
