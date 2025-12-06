# AIRR-ML-25: Adaptive Immune Profiling Challenge - Submissions

## Submission Requirements

### File Format

- **Filename**: `submission.csv`
- **Format**: CSV (Comma-Separated Values)
- **Encoding**: UTF-8

### Submission Structure

The submission file must contain predictions for **all test repertoires** across all datasets.

| Column | Type | Description |
|--------|------|-------------|
| `repertoire_id` | string | Unique identifier matching test metadata |
| `label_positive_pred` | float | Probability (0-1) that repertoire is label-positive |
| `junction_aa` | string | Label-associated amino acid sequences (comma-separated) |

### Required Rows

- **Total rows required**: 404,213 rows
- **Test repertoires**: 4,213 repertoires across 4 test datasets
- Must include predictions for ALL test repertoires

---

## Submission Format Example

```csv
repertoire_id,label_positive_pred,junction_aa
rep_001,0.85,"CASSLGQAYEQYF,CASSYSGGNTGELFF,CASSPRDRADEQFF"
rep_002,0.12,"CASSLAGTYEQYF"
rep_003,0.67,"CASSLDRAGYNEQFF,CASSQETQYF"
...
```

### Field Details

#### `repertoire_id`
- Must exactly match the `repertoire_id` values in the test metadata
- One row per repertoire

#### `label_positive_pred`
- **Range**: 0.0 to 1.0
- Represents the predicted probability that the repertoire is label-positive
- Used for **Task 1** evaluation (AUC-ROC)

#### `junction_aa`
- Comma-separated list of amino acid sequences
- These are the sequences predicted to be associated with the label
- Used for **Task 2** evaluation (Jaccard Similarity)
- Can be empty string if no sequences predicted

---

## Submission Limits

### Daily Limits
- **5 submissions per day** per team

### Final Submission Limits
- **Maximum 2 final submissions** selected for final scoring

### Team Limits
- **Maximum 5 team members** per team
- Cannot submit from multiple accounts

---

## Evaluation Process

### Public Leaderboard
- Scores calculated on a **subset** of test data
- Updated in real-time as submissions are made
- Used for preliminary ranking during competition

### Private Leaderboard
- Scores calculated on the **remaining** test data
- Revealed only after competition ends
- Determines **final rankings and prize winners**

### Scoring Metric
The final score is a **weighted average** of two metrics:

1. **AUC-ROC (Area Under ROC Curve)**
   - Evaluates Task 1: Classification of label-positive repertoires
   - Measures how well the model discriminates between positive and negative repertoires

2. **Jaccard Similarity**
   - Evaluates Task 2: Identification of label-associated sequences
   - Measures overlap between predicted and true label-associated sequences
   - Formula: |A ∩ B| / |A ∪ B|

---

## Submission Timeline

| Date | Event |
|------|-------|
| October 9, 2025 | Competition Start |
| December 10, 2025 | Entry Deadline |
| December 10, 2025 | Team Merger Deadline |
| **December 17, 2025** | **Final Submission Deadline (06:59:59 UTC)** |

---

## Code Requirements for Winners

### Mandatory for Prize Money

Winners must submit their code following the official template:

1. **Fork the official repository**: https://github.com/uio-bmi/predict-airr

2. **Implement the `ImmuneStatePredictor` class** in `predictor.py`:
   ```python
   class ImmuneStatePredictor(BasePredictor):
       def __init__(self):
           pass

       def train(self, train_data_path: str, n_jobs: int, device: str):
           """Train the model on provided data"""
           pass

       def predict(self, test_data_path: str, output_path: str):
           """Generate predictions for test data"""
           pass
   ```

3. **Command-line interface**:
   ```bash
   python3 -m submission.main \
       --train_dir /path/to/train_dir \
       --test_dir /path/to/test_dir \
       --out_dir /path/to/output_dir \
       --n_jobs 4 \
       --device cpu
   ```

4. **Update `requirements.txt`** with exact version pins

5. **License**: Code must be released under **MIT Open Source License**

---

## Submission Validation

### Common Errors

1. **Missing repertoire_id**: All test repertoires must have predictions
2. **Invalid probability values**: Must be between 0 and 1
3. **Incorrect column names**: Must exactly match required column names
4. **Wrong number of rows**: Must have exactly 404,213 rows
5. **Encoding issues**: Use UTF-8 encoding

### Validation Checklist

- [ ] File is named `submission.csv`
- [ ] Contains all three required columns
- [ ] All 404,213 rows present
- [ ] `label_positive_pred` values are floats between 0-1
- [ ] `repertoire_id` values match test metadata exactly
- [ ] `junction_aa` sequences are valid amino acid strings
- [ ] File is UTF-8 encoded

---

## Test Datasets

Submissions must include predictions for repertoires from all test datasets:

| Dataset | Description |
|---------|-------------|
| test_dataset_1 | Test data (subset of training distribution) |
| test_dataset_2 | Test data (subset of training distribution) |
| test_dataset_3 | Test data (subset of training distribution) |
| test_dataset_4 | Test data (subset of training distribution) |

**Note**: Test labels are NOT provided. Predictions are evaluated by Kaggle's scoring system.

---

## Submission Strategies

### Best Practices

1. **Start with baseline**: Use the official Example Baseline Predictor notebook
2. **Validate locally**: Check submission format before uploading
3. **Monitor leaderboard**: Track score changes across submissions
4. **Diversify approaches**: Try different models for ensemble
5. **Save submissions**: Keep track of best-performing submissions

### Final Submission Selection

- You can select up to **2 submissions** for final evaluation
- Choose submissions that:
  - Perform well on public leaderboard
  - Use different approaches (for diversity)
  - Are stable across validation folds

---

## Technical Notes

### Large Dataset Handling

- Total dataset size: **19.94 GB**
- Consider using:
  - Data sampling for initial experiments
  - Parquet format for efficient storage (see Jirka Borovec's notebook)
  - Batch processing for memory efficiency

### Computational Resources

- Kaggle provides **free GPU** (limited hours)
- Consider cloud resources for intensive training
- Optimize code for efficiency

---

## Post-Competition Requirements

### For Prize Winners (Top 3)

1. **Open-source code**: Must release under MIT license
2. **Reproducibility**: Code must produce submitted results
3. **Documentation**: Include README with instructions

### For Top 10 Teams

- **Scientific paper invitation**: Contribute to Nature Methods publication
- **Paper is "accepted in principle"**
- Opportunity to be co-authors on peer-reviewed publication

---

## Support and Resources

### Official Resources
- **Code Template**: https://www.kaggle.com/code/ckanduri/code-template
- **Baseline Predictor**: https://www.kaggle.com/code/ckanduri/example-baseline-predictor-using-code-template
- **GitHub Repository**: https://github.com/uio-bmi/predict-airr

### Community Resources
- **Discussion Forum**: For questions and clarifications
- **Public Notebooks**: Learn from other participants' approaches

### Host Contact
- Competition Host: **Chakravarthi Kanduri**
- Very active in discussion forum (93% response rate)

---

## Quick Reference

| Item | Value |
|------|-------|
| File Name | `submission.csv` |
| Total Rows | 404,213 |
| Daily Limit | 5 submissions |
| Final Selections | 2 submissions |
| Deadline | Dec 17, 2025 (06:59:59 UTC) |
| Prize Pool | $10,000 |
| Top Prize | $5,000 |

