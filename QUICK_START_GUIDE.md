# Quick Start Guide - Class Imbalance Handling

This guide shows you how to use the improved `main_gaze.py` with class imbalance handling.

## What Changed?

Your training script now includes advanced techniques to handle class imbalance:

1. ✅ **Batch Balancing** - Minority samples appear more frequently
2. ✅ **Per-Class F1 Tracking** - See performance for each class
3. ✅ **Label Smoothing** - Reduces majority class overconfidence
4. ✅ **Macro-F1 Early Stopping** - Better model selection
5. ✅ **Comprehensive Metrics** - Detailed performance analysis

## Running the Script

Simply run the script as before:

```bash
python main_gaze.py
```

The improvements are **automatically enabled** with sensible defaults.

## Customizing Settings

To adjust the configuration, edit the **CONFIGURATION SECTION** at the top of `main()`:

```python
# ========================================================================
# CONFIGURATION SECTION - Adjust these parameters for your dataset
# ========================================================================

# Training hyperparameters
BATCH_SIZE = 16              # Batch size
EPOCHS = 30                  # Training epochs (increased for minority learning)
LEARNING_RATE = 5e-5         # Smaller LR for stable convergence
ACCUM_ITER = 2

# Class imbalance handling
USE_BALANCED_SAMPLING = True  # Enable batch balancing
MINORITY_PER_BATCH = 2        # Target minority samples per batch
USE_LABEL_SMOOTHING = True    # Enable label smoothing
LABEL_SMOOTHING = 0.1         # Smoothing factor (0.1 = 10% smoothing)
USE_MACRO_F1_STOPPING = True  # Use macro-F1 instead of accuracy

# Early stopping and scheduling
EARLY_STOP_PATIENCE = 10      # Epochs without improvement before stopping
SCHEDULER_PATIENCE = 5        # Epochs before reducing learning rate
SCHEDULER_FACTOR = 0.5        # LR reduction factor

# Diagnostic configuration
DIAGNOSTIC_FREQUENCY = 10     # Full analysis every N epochs (0 to disable)
```

## Understanding the Output

### During Training

You'll see detailed metrics for each epoch:

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
  
  Confusion Matrix:
[[245  35]    <- Class 0: 245 correct, 35 misclassified as class 1
 [ 42  28]]   <- Class 1: 28 correct, 42 misclassified as class 0
```

### Key Metrics to Watch

- **Macro-F1**: Average of per-class F1 scores (treats all classes equally)
- **Per-class F1**: F1 score for each class individually
- **Confusion Matrix**: Shows which classes are being confused

### Good vs Bad Performance

**Good** (balanced performance):
```
Per-class F1 scores: [0.85 0.82]  # Both classes performing well
Macro-F1: 0.835
```

**Bad** (imbalanced, majority bias):
```
Per-class F1 scores: [0.95 0.10]  # Minority class performing poorly
Macro-F1: 0.525                    # Low despite high accuracy
```

## Common Adjustments

### If Minority Class F1 is Still Low

1. **Increase batch balancing**:
   ```python
   MINORITY_PER_BATCH = 3  # or 4
   ```

2. **Increase label smoothing**:
   ```python
   LABEL_SMOOTHING = 0.15  # or 0.2
   ```

3. **Further reduce learning rate**:
   ```python
   LEARNING_RATE = 2e-5
   ```

### If Training is Too Slow

1. **Reduce epochs**:
   ```python
   EPOCHS = 20
   ```

2. **Reduce diagnostic frequency**:
   ```python
   DIAGNOSTIC_FREQUENCY = 20  # or 0 to disable
   ```

3. **Increase batch size** (if GPU memory allows):
   ```python
   BATCH_SIZE = 32
   ```

### If You Don't Want Class Imbalance Handling

Simply set these to False/disabled:

```python
USE_BALANCED_SAMPLING = False
USE_LABEL_SMOOTHING = False
USE_MACRO_F1_STOPPING = False  # Will use accuracy instead
```

## Saved Model

The best model is saved to `best_model_gaze_attention_fixed.pth` and includes:

- Model weights
- Optimizer state
- Best accuracy and macro-F1 scores
- Per-class F1 scores
- Confusion matrix
- Training history

To load the model:

```python
checkpoint = torch.load('best_model_gaze_attention_fixed.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Best Accuracy: {checkpoint['accuracy']:.2f}%")
print(f"Best Macro-F1: {checkpoint['macro_f1']:.4f}")
print(f"Per-class F1: {checkpoint['per_class_f1']}")
```

## Training History

Training history is saved to `training_history_fixed.npy`:

```python
import numpy as np
history = np.load('training_history_fixed.npy', allow_pickle=True).item()

# Available keys:
# - train_loss, train_acc, train_macro_f1, train_f1_per_class
# - eval_acc, eval_macro_f1, eval_f1_per_class
# - eval_precision_per_class, eval_recall_per_class

# Plot learning curves
import matplotlib.pyplot as plt
plt.plot(history['train_macro_f1'], label='Train Macro-F1')
plt.plot(history['eval_macro_f1'], label='Val Macro-F1')
plt.legend()
plt.show()
```

## Troubleshooting

### "No module named 'sklearn'"

The script will work fine without sklearn - it uses fallback implementations. But if you want to install it:

```bash
pip install scikit-learn
```

### "Target contains invalid class index"

This means your labels are outside the expected range. Check:
- Are labels 0 and 1 (for binary classification)?
- Any NaN or corrupted label values?
- Correct number of classes in model configuration?

### Validation Accuracy is 0%

This is expected at the start. The script will run diagnostics automatically. If it persists:
- Check data loading is working correctly
- Verify EEG and gaze files are matched properly
- Ensure model architecture matches your data dimensions

## Getting Help

See `CLASS_IMBALANCE_IMPROVEMENTS.md` for detailed technical documentation.

For issues:
1. Check the console output for error messages
2. Review the confusion matrix to understand model behavior
3. Try adjusting configuration parameters
4. Verify your data is loaded correctly (check diagnostics)

## Expected Results

With these improvements, you should see:

✓ **Minority class F1 increases** over epochs
✓ **More balanced confusion matrix** (not all predictions to majority class)
✓ **Macro-F1 correlates better** with actual model quality
✓ **Better model generalization** to real-world data

Happy training! 🚀
