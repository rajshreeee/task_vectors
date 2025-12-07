from src.args import parse_arguments
from src.finetune import finetune

# Fine-tune on IMBALANCED version (BI-RADS 4 is minority with 200 samples)
class ArgsImbalanced:
    data_location = '/scratch/project_2016790/task_vectors/data'
    train_dataset = 'BIRADSImbalanced'  # Imbalanced version
    eval_datasets = ['BIRADSImbalanced']
    model = 'ViT-B-32'
    batch_size = 16
    lr = 1e-5
    wd = 0.1
    ls = 0.0
    warmup_length = 100
    epochs = 5
    save = '/scratch/project_2016790/checkpoints/ViT-B-32/imbalanced'
    cache_dir = None
    openclip_cachedir = '/scratch/project_2016790/cache/open_clip'
    device = 'cuda'
    load = None
    results_db = '/scratch/project_2016790/checkpoints/results_imbalanced.jsonl'
    exp_name = 'birads_imbalanced'

args_imbalanced = ArgsImbalanced()
zs_path_imbalanced, ft_path_imbalanced = finetune(args_imbalanced)

print("\n" + "="*70)
print("IMBALANCED VERSION COMPLETE")
print("="*70)