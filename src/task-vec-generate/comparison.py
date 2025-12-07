import torch
from src.args import parse_arguments
from src.eval import eval_single_dataset

# ============================================================================
# SIMPLE EVALUATION
# ============================================================================

base_dir = '/scratch/project_2016790/checkpoints/ViT-B-32'

# Parse args properly
args = parse_arguments()
args.data_location = '/scratch/project_2016790/task_vectors/data'
args.batch_size = 32
args.device = 'cuda'
args.save = '/scratch/project_2016790/checkpoints/ViT-B-32/imbalanced'

# Evaluate baseline
print("\nBaseline (Imbalanced):")
ft_encoder = torch.load(f'{base_dir}/imbalanced/BIRADSImbalanced/finetuned.pt', weights_only=False)
baseline = eval_single_dataset(ft_encoder, 'BIRADSImbalanced', args)

# Evaluate enhanced models
alphas = [0.0, 0.3, 0.5, 0.7, 1.0, 1.5]
results = {}

for alpha in alphas:
    print(f"\nAlpha {alpha}:")
    encoder = torch.load(f'{base_dir}/enhanced_alpha_{alpha}.pt', weights_only=False)
    results[alpha] = eval_single_dataset(encoder, 'BIRADSImbalanced', args)

# Print comparison
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Baseline: {baseline['top1']:.4f}")
for alpha, res in results.items():
    improvement = res['top1'] - baseline['top1']
    print(f"Alpha {alpha}: {res['top1']:.4f} ({improvement:+.4f})")
