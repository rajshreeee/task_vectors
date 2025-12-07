import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from src.args import parse_arguments
from src.modeling import ImageClassifier
from src.datasets.registry import get_dataset
from src.datasets.common import get_dataloader, maybe_dictionarize
import tqdm

# ============================================================================
# ENHANCED EVALUATION WITH CONFUSION MATRIX
# ============================================================================

def eval_with_confusion_matrix(model, dataset_name, args):
    """Evaluate with confusion matrix and per-class accuracy
    
    Args:
        model: Complete ImageClassifier (encoder + head)
        dataset_name: Name of dataset to evaluate on
        args: Arguments with data_location, batch_size, device
    """
    
    model.eval()
    model = model.to(args.device)
    
    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader, desc=f"Evaluating"):
            data = maybe_dictionarize(data)
            x = data['images'].to(args.device)
            y = data['labels'].to(args.device)
            
            logits = model(x)
            pred = logits.argmax(dim=1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Overall accuracy
    accuracy = (all_preds == all_labels).mean()
    
    # Per-class accuracy
    classnames = dataset.classnames
    per_class_acc = {}
    per_class_count = {}
    
    for i, class_name in enumerate(classnames):
        mask = all_labels == i
        count = mask.sum()
        per_class_count[class_name] = int(count)
        if count > 0:
            per_class_acc[class_name] = float((all_preds[mask] == all_labels[mask]).mean())
        else:
            per_class_acc[class_name] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'top1': accuracy,
        'per_class_acc': per_class_acc,
        'per_class_count': per_class_count,
        'confusion_matrix': cm,
        'classnames': classnames
    }


# ============================================================================
# CONFIGURATION
# ============================================================================

base_dir = '/scratch/project_2016790/checkpoints/ViT-B-32'

# Parse args
args = parse_arguments()
args.data_location = '/scratch/project_2016790/task_vectors/data'
args.batch_size = 32
args.device = 'cuda'

# ============================================================================
# LOAD BASELINE MODEL (Complete model with encoder + head)
# ============================================================================

print("\n" + "="*70)
print("LOADING MODELS")
print("="*70)

# Load the complete ImageClassifier (not just encoder)
# The finetuned model should have both encoder and classification head
baseline_model_path = f'{base_dir}/imbalanced/BIRADSImbalanced/finetuned.pt'

print(f"Loading baseline from: {baseline_model_path}")

# Try to load as complete model first
try:
    baseline_model = torch.load(baseline_model_path, weights_only=False)
    print(f"Loaded model type: {type(baseline_model)}")
    
    # Check if it's an ImageEncoder or ImageClassifier
    if not hasattr(baseline_model, 'classification_head'):
        print("Model is ImageEncoder only, need to create classification head")
        # We need to recreate the classification head
        from src.heads import get_classification_head
        
        # Create args for head creation
        class HeadArgs:
            save = '/scratch/project_2016790/checkpoints/ViT-B-32/imbalanced'
            model = 'ViT-B-32'
            data_location = args.data_location
            openclip_cachedir = '/scratch/project_2016790/cache/open_clip'
        
        head_args = HeadArgs()
        classification_head = get_classification_head(head_args, 'BIRADSImbalanced')
        baseline_model = ImageClassifier(baseline_model, classification_head)
        print("Created ImageClassifier with classification head")
    else:
        print("Model already has classification head")
        
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# ============================================================================
# EVALUATE BASELINE
# ============================================================================

print("\n" + "="*70)
print("BASELINE EVALUATION")
print("="*70)

baseline = eval_with_confusion_matrix(baseline_model, 'BIRADSBalancedTest', args)

print(f"\nOverall Accuracy: {baseline['top1']:.4f} ({100*baseline['top1']:.2f}%)")
print("\nPer-class Accuracy:")
for class_name in baseline['classnames']:
    acc = baseline['per_class_acc'][class_name]
    count = baseline['per_class_count'][class_name]
    print(f"  {class_name}: {acc:.4f} ({count} samples)")

print("\nConfusion Matrix:")
print("Rows: True labels, Columns: Predictions")
print(f"Classes: {baseline['classnames']}")
print(baseline['confusion_matrix'])

# ============================================================================
# EVALUATE ENHANCED MODELS
# ============================================================================

print("\n" + "="*70)
print("ENHANCED MODELS EVALUATION")
print("="*70)

alphas = [0.0, 0.3, 0.5, 0.7, 1.0, 1.5]
results = {}

# Get the classification head from baseline to reuse
classification_head = baseline_model.classification_head

for alpha in alphas:
    print(f"\n--- Alpha {alpha} ---")
    
    # Load enhanced encoder
    enhanced_encoder = torch.load(f'{base_dir}/enhanced_alpha_{alpha}.pt', weights_only=False)
    
    # Create ImageClassifier with enhanced encoder + same classification head
    enhanced_model = ImageClassifier(enhanced_encoder, classification_head)
    
    results[alpha] = eval_with_confusion_matrix(enhanced_model, 'BIRADSBalancedTest', args)
    
    print(f"Overall Accuracy: {results[alpha]['top1']:.4f}")
    print("Per-class Accuracy:")
    for class_name in results[alpha]['classnames']:
        acc = results[alpha]['per_class_acc'][class_name]
        print(f"  {class_name}: {acc:.4f}")

# ============================================================================
# COMPARISON TABLES
# ============================================================================

print("\n" + "="*70)
print("OVERALL ACCURACY COMPARISON")
print("="*70)
print(f"{'Model':<20} {'Accuracy':<12} {'Improvement':<12}")
print("-" * 70)
print(f"{'Baseline':<20} {baseline['top1']:.4f}       -")
for alpha in alphas:
    improvement = results[alpha]['top1'] - baseline['top1']
    print(f"{'Alpha ' + str(alpha):<20} {results[alpha]['top1']:.4f}       {improvement:+.4f}")

print("\n" + "="*70)
print("PER-CLASS ACCURACY COMPARISON")
print("="*70)

# Header
header = f"{'Model':<20}"
for cn in baseline['classnames']:
    header += f"{cn:<18}"
print(header)
print("-" * 100)

# Baseline
row = f"{'Baseline':<20}"
for cn in baseline['classnames']:
    row += f"{baseline['per_class_acc'][cn]:.4f}             "
print(row)

# Enhanced
for alpha in alphas:
    row = f"{'Alpha ' + str(alpha):<20}"
    for cn in baseline['classnames']:
        row += f"{results[alpha]['per_class_acc'][cn]:.4f}             "
    print(row)

print("\n" + "="*70)
print("PER-CLASS IMPROVEMENT")
print("="*70)

header = f"{'Model':<20}"
for cn in baseline['classnames']:
    header += f"{cn:<18}"
print(header)
print("-" * 100)

for alpha in alphas:
    row = f"{'Alpha ' + str(alpha):<20}"
    for cn in baseline['classnames']:
        improvement = results[alpha]['per_class_acc'][cn] - baseline['per_class_acc'][cn]
        row += f"{improvement:+.4f}            "
    print(row)

# ============================================================================
# BEST ALPHA
# ============================================================================

best_overall = max(alphas, key=lambda a: results[a]['top1'])
best_birads5 = max(alphas, key=lambda a: results[a]['per_class_acc']['BI-RADS 5'])

print("\n" + "="*70)
print("BEST ALPHA")
print("="*70)
print(f"\nBest Overall: α={best_overall}")
print(f"  Accuracy: {results[best_overall]['top1']:.4f}")
print(f"  Improvement: {results[best_overall]['top1'] - baseline['top1']:+.4f}")

print(f"\nBest BI-RADS 5 (minority class): α={best_birads5}")
print(f"  BI-RADS 5 Accuracy: {results[best_birads5]['per_class_acc']['BI-RADS 5']:.4f}")
print(f"  Improvement: {results[best_birads5]['per_class_acc']['BI-RADS 5'] - baseline['per_class_acc']['BI-RADS 5']:+.4f}")

# ============================================================================
# CONFUSION MATRICES
# ============================================================================

print("\n" + "="*70)
print("CONFUSION MATRICES")
print("="*70)

print(f"\nBaseline:")
print(baseline['confusion_matrix'])

print(f"\nBest Overall (α={best_overall}):")
print(results[best_overall]['confusion_matrix'])

print(f"\nBest BI-RADS 5 (α={best_birads5}):")
print(results[best_birads5]['confusion_matrix'])

print("\n" + "="*70)
print("EVALUATION COMPLETE!")
print("="*70)
