import sys
from src.args import parse_arguments
from src.finetune import finetune


# Get focus class from command line (default to 2)
if len(sys.argv) > 1:
    focus_class = sys.argv[1]
else:
    focus_class = '2'  # Default


# Configuration based on focus class
configs = {
    '2': {
        'dataset': 'BIRADSDensity2Focus',
        'save_dir': 'birads-2-focus',
        'results_db': 'results_birads_2_focus.jsonl',
        'exp_name': 'birads_2_focus',
        'name': 'BIRADS 2 FOCUS'
    },
    '5': {
        'dataset': 'BIRADSDensity5Focus',
        'save_dir': 'birads-5-focus',
        'results_db': 'results_birads_5_focus.jsonl',
        'exp_name': 'birads_5_focus',
        'name': 'BIRADS 5 FOCUS'
    }
}

if focus_class not in configs:
    print(f"Error: Invalid focus class '{focus_class}'. Use '2' or '5'.")
    print("Usage: python train_focus.py [2|5]")
    sys.exit(1)

config = configs[focus_class]


# Create args
class ArgsFocus:
    data_location = '/scratch/project_2016790/task_vectors/data'
    train_dataset = config['dataset']
    eval_datasets = [config['dataset']]
    model = 'ViT-B-32'
    batch_size = 16
    lr = 1e-5
    wd = 0.1
    ls = 0.0
    warmup_length = 50
    epochs = 5
    save = f"/scratch/project_2016790/checkpoints/ViT-B-32/{config['save_dir']}"
    cache_dir = None
    openclip_cachedir = '/scratch/project_2016790/cache/open_clip'
    device = 'cuda'
    load = None
    results_db = f"/scratch/project_2016790/checkpoints/{config['results_db']}"
    exp_name = config['exp_name']


print("\n" + "="*70)
print(f"FINE-TUNING {config['name']}")
print("="*70)

args_focus = ArgsFocus()
zs_path_focus, ft_path_focus = finetune(args_focus)

print("\n" + "="*70)
print(f"{config['name']} COMPLETE")
print("="*70)
print(f"Zero-shot: {zs_path_focus}")
print(f"Fine-tuned: {ft_path_focus}")
print("="*70)
