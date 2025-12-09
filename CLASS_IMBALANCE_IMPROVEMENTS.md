# Class Imbalance Improvements for main_gaze.py

This document describes the improvements made to `main_gaze.py` to address severe class imbalance issues that were causing poor training results.

## Problem Statement

The original implementation suffered from severe class imbalance, leading to:
- Poor minority class performance
- Model bias towards majority class
- Misleading accuracy metrics
- Overfitting on majority class

## Implemented Solutions

### 1. Batch Balancing with WeightedRandomSampler

**What it does**: Ensures each batch contains minority class samples by oversampling the minority class during batch creation.

**Configuration**:
```python
use_balanced_sampling = True  # Enable batch balancing
minority_per_batch = 2  # Minimum minority samples per batch
```

**How it works**:
- Computes sample weights inversely proportional to class frequency
- Uses `WeightedRandomSampler` to oversample minority class
- Automatically balances batches without data duplication on disk

**Benefits**:
- Guarantees minority class exposure in every batch
- Prevents majority class domination
- No manual data augmentation required

### 2. Per-Class F1 Score Tracking

**What it does**: Tracks precision, recall, and F1 score for each class during training and evaluation.

**Metrics computed**:
- Per-class Precision: How many predicted positives are actually positive
- Per-class Recall: How many actual positives are correctly identified
- Per-class F1: Harmonic mean of precision and recall
- Macro-F1: Average F1 across all classes (treats all classes equally)
- Confusion Matrix: Shows true vs predicted labels

**Benefits**:
- Identifies which class is underperforming
- Monitors minority class learning progress
- More informative than accuracy alone

### 3. Label Smoothing for Majority Class

**What it does**: Reduces model overconfidence by distributing some probability mass from the target class to other classes.

**Configuration**:
```python
use_label_smoothing = True  # Enable label smoothing
label_smoothing = 0.1  # Smoothing factor (0.0 to 1.0)
```

**How it works**:
- Instead of hard labels [0, 1] or [1, 0]
- Uses soft labels like [0.05, 0.95] or [0.95, 0.05]
- Prevents the model from becoming too confident in its predictions

**Benefits**:
- Reduces overfitting on majority class
- Improves generalization
- Better calibrated probabilities

### 4. Optimized Learning Rate Schedule

**What it does**: Uses a smaller learning rate with longer training for better minority class learning.

**Configuration**:
```python
lr = 5e-5  # Smaller learning rate (was 1e-4)
epochs = 30  # Longer training (was 15)
patience = 5  # Increased scheduler patience (was 2)
factor = 0.5  # Gentler LR reduction (was 0.1)
```

**Benefits**:
- More stable convergence
- Better fine-tuning for minority class
- Prevents overshooting optimal weights

### 5. Macro-F1 Early Stopping

**What it does**: Uses macro-F1 score instead of accuracy for model selection and early stopping.

**Configuration**:
```python
use_macro_f1_stopping = True  # Use macro-F1 instead of accuracy
early_stop_patience = 10  # Epochs to wait before stopping
```

**Why macro-F1 is better**:
- **Accuracy**: Can be high even with poor minority class performance
  - Example: 95% accuracy with 100% majority, 0% minority
- **Macro-F1**: Treats all classes equally, exposes imbalance
  - Example: 50% macro-F1 with same predictions (average of 100% and 0%)

**Benefits**:
- Selects models that perform well on ALL classes
- Prevents majority class bias
- More reliable model selection

### 6. Class Weights in Loss Function

**What it does**: Weights the loss function to penalize minority class errors more heavily.

**How it works**:
```python
# Automatically computed from training data
class_weights = compute_class_weights(all_train_labels, device)
# Example output: [0.6, 2.4] for 80/20 imbalance
```

**Benefits**:
- Forces model to pay attention to minority class
- Balances gradient contributions
- Compensates for class imbalance in loss

### 7. Comprehensive Evaluation Metrics

**What's tracked**:
- Training: loss, accuracy, macro-F1, per-class F1
- Validation: accuracy, macro-F1, per-class F1, precision, recall
- Confusion matrix at each epoch
- Best model saved based on macro-F1

**Output format**:
```
EPOCH 5 RESULTS
================================================================================

TRAINING:
  Loss: 0.4532 | Accuracy: 78.50% | Macro-F1: 0.6234
  Per-class F1 scores: [0.8421 0.4047]
  Per-class Precision: [0.8234 0.5123]
  Per-class Recall: [0.8611 0.3345]

VALIDATION:
  Accuracy: 76.20% | Macro-F1: 0.5890
  Per-class F1 scores: [0.8156 0.3624]
  Per-class Precision: [0.7989 0.4123]
  Per-class Recall: [0.8328 0.3234]

  Confusion Matrix:
[[245  35]
 [ 42  28]]
```

## Usage

The improvements are automatically enabled in `main_gaze.py`. To customize:

```python
# In main() function, adjust these parameters:

# Batch balancing
use_balanced_sampling = True  # Set to False to disable
minority_per_batch = 2  # Adjust based on batch size

# Label smoothing
use_label_smoothing = True  # Set to False to disable
label_smoothing = 0.1  # Range: 0.0 (none) to 0.5 (aggressive)

# Early stopping metric
use_macro_f1_stopping = True  # Set to False to use accuracy

# Training parameters
batch_size = 16
epochs = 30
lr = 5e-5
```

## Expected Results

With these improvements, you should see:

1. **Better Minority Class Performance**
   - Minority class F1 score significantly improved
   - More balanced confusion matrix
   - Better recall on minority class

2. **More Reliable Model Selection**
   - Best model based on balanced performance
   - Not biased towards majority class
   - Better generalization

3. **Clearer Training Progress**
   - Per-class metrics show which class is improving
   - Early detection of overfitting on majority class
   - Better understanding of model behavior

## Troubleshooting

### Issue: Model still biased towards majority class
**Solutions**:
- Increase `minority_per_batch` (e.g., to 3-4)
- Increase `label_smoothing` (e.g., to 0.15-0.2)
- Decrease learning rate further (e.g., to 2e-5)
- Check class weights are being applied correctly

### Issue: Training is too slow
**Solutions**:
- Reduce `epochs` (but keep > 20)
- Increase `batch_size` if memory allows
- Reduce `early_stop_patience` (but keep >= 5)

### Issue: Minority class still has low F1
**Solutions**:
- Collect more minority class data if possible
- Increase class weight for minority class manually
- Use data augmentation for minority class
- Try different architectures

## Dependencies

- PyTorch (existing)
- NumPy (existing)
- tqdm (existing)
- scikit-learn (optional - fallback implementations provided)

If scikit-learn is not available, the code will use manual implementations of metrics without any loss of functionality.

## Technical Notes

### WeightedRandomSampler
- Samples with replacement, so some samples may appear multiple times per epoch
- This is intentional and helps balance the batches
- Total samples per epoch = len(dataset)

### Label Smoothing
- Applied only when `use_label_smoothing=True`
- Works with class weights (both can be used together)
- Smoothing factor typically 0.1 for binary classification

### Macro-F1 vs Micro-F1
- We use **Macro-F1**: average of per-class F1 (treats classes equally)
- Micro-F1 would be equivalent to accuracy (not useful for imbalanced data)

## References

1. **Label Smoothing**: "Rethinking the Inception Architecture for Computer Vision" (Szegedy et al., 2016)
2. **Class Imbalance**: "A systematic study of the class imbalance problem in convolutional neural networks" (Buda et al., 2018)
3. **F1 Score**: Standard machine learning metric, well-suited for imbalanced datasets

## History

- **v1.0** (2024): Initial implementation with basic training
- **v2.0** (Current): Added comprehensive class imbalance handling
  - Batch balancing
  - Per-class F1 tracking
  - Label smoothing
  - Optimized learning schedule
  - Macro-F1 early stopping
  - Comprehensive metrics
