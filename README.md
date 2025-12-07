# Task Vector Experiments

This repository explores the application of task arithmetic through various experiments which was introduced in [EDITING MODELS WITH TASK ARITHMETIC](https://arxiv.org/pdf/2212.04089)

> **Forked from:** [mlfoundations/task_vectors](https://github.com/mlfoundations/task_vectors)  
> **Experiment Status:** Ongoing with mixed results

## Experiment 1: Task Vectors for BIRADS Class Imbalance 

Medical imaging datasets often suffer from severe class imbalance, particularly for rare categories. In BI-RADS classification, malignant cases (BI-RADS 5) are significantly underrepresented, leading to poor model performance. This work tests whether task vector analogies can transfer knowledge to improve minority class detection.

## Experimental Setup

### Datasets
[Mini DDSM](https://www.kaggle.com/datasets/cheddad/miniddsm2/discussion)

**1. Baseline (Imbalanced Multi-class)**
- BI-RADS 2, 3, 4: 800 samples each
- BI-RADS 5: 300 samples (minority class)
- 4-class classification task

**2. BI-RADS 2 Focus (Binary)**
- 300 BI-RADS 2 vs. 300 non-BI-RADS 2 samples
- Accuracy: 74.17%

**3. BI-RADS 5 Focus (Binary)**
- 300 BI-RADS 5 vs. 300 non-BI-RADS 5 samples  
- Accuracy: 80.83%

### Technical Details

- **Model**: OpenCLIP ViT-B-32 pre-trained
- **Framework**: PyTorch
- **Training**: 5 epochs, batch size 16
- **Evaluation**: Balanced test set (100 samples per class)

### Task Arithmetic Approach


An enhancement vector was created to capture the "malignancy shift"


```
Enhancement Vector = TaskVector(BI-RADS 5 focus) - TaskVector(BI-RADS 2 focus)
Enhanced Model = Base Model + α × Enhancement Vector
```


## Results


- **Minority class improvement**: BI-RADS 5 accuracy increased from 35% → 55% at α=1.0 (+20 percentage points) 
- **Accuracy Decline In Control Task**: BI-RADS 3 and 4 accuracy degraded as α increased

### Confusion Matrix Comparison

**Baseline** (Overall: 53.25%)
```
Pred:       2   3   4   5
True 2:    71  24   5   0
True 3:    33  59   6   2
True 4:     7  37  48   8
True 5:     8  18  39  35
```

**Best for BI-RADS 5** (α=1.0, Overall: 47.75%)
```
Pred:       2   3   4   5
True 2:    78  18   2   2
True 3:    46  46   2   6
True 4:    12  55  14  19
True 5:    13  21  11  55
```

