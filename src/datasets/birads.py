import os
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter


def load_birads_data(location):
    """Load and filter BI-RADS data to densities 1-4 (BI-RADS 2-5)"""
    data_dir = Path(location) / 'MINI-DDSM-Complete-JPEG-8'
    metadata_file = data_dir / 'DataWMask.xlsx'
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_file}")
    
    df_meta = pd.read_excel(metadata_file)
    
    # Map images to BI-RADS categories
    data = []
    class_folders = {
        'Normal': data_dir / 'Normal',
        'Benign': data_dir / 'Benign',
        'Cancer': data_dir / 'Cancer'
    }
    
    for class_name, class_path in class_folders.items():
        if not class_path.exists():
            continue
        
        case_folders = [f for f in class_path.iterdir() if f.is_dir()]
        
        for case_folder in case_folders:
            case_id = case_folder.name
            images = [img for img in case_folder.glob('*.jpg') 
                     if 'mask' not in img.name.lower() and 'boundary' not in img.name.lower()]
            
            for img_path in images:
                matching_rows = df_meta[df_meta['fileName'].str.contains(case_id, na=False)]
                if len(matching_rows) == 0:
                    matching_rows = df_meta[df_meta['fullPath'].str.contains(case_id, na=False)]
                
                if len(matching_rows) > 0:
                    row = matching_rows.iloc[0]
                    birads = int(row['Density'])
                    data.append({
                        'image_path': str(img_path),
                        'birads': birads,
                        'classification': class_name,
                    })
    
    df = pd.DataFrame(data)
    
    # Filter to densities 1-4 (BI-RADS 2-5), skip 0
    df = df[df['birads'].isin([1, 2, 3, 4])].copy()
    
    print(f"Loaded BI-RADS data:")
    print(df['birads'].value_counts().sort_index())
    
    return df


def sample_distribution(df, sample_config, seed=42):
    """
    Sample data according to configuration.
    
    Args:
        df: DataFrame with 'birads' column
        sample_config: Dict mapping birads value -> number of samples
        seed: Random seed
    
    Returns:
        Sampled DataFrame
    """
    dfs = []
    for birads_val, n_samples in sample_config.items():
        df_class = df[df['birads'] == birads_val]
        n_available = len(df_class)
        n_to_sample = min(n_available, n_samples)
        
        if n_to_sample < n_samples:
            print(f"  Warning: Density {birads_val} has only {n_available} samples, requested {n_samples}")
        
        df_sampled = df_class.sample(n=n_to_sample, random_state=seed)
        dfs.append(df_sampled)
        print(f"  Density {birads_val} (BI-RADS {birads_val + 1}): {n_to_sample} samples")
    
    df_result = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df_result


# ============================================================================
# MULTI-CLASS DATASET
# ============================================================================

class MultiClassBIRADSDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for multi-class BI-RADS classification"""
    
    def __init__(self, data_list, preprocess):
        self.data = data_list
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        try:
            image = Image.open(item['image_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading {item['image_path']}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply CLIP preprocessing
        if self.preprocess is not None:
            image = self.preprocess(image)
        
        # Multi-class label
        # Density 1 -> label 0 (BI-RADS 2)
        # Density 2 -> label 1 (BI-RADS 3)
        # Density 3 -> label 2 (BI-RADS 4)
        # Density 4 -> label 3 (BI-RADS 5)
        label = item['birads'] - 1
        
        return image, label


class BIRADSImbalanced:
    """
    Multi-class imbalanced dataset: BI-RADS 5 (Density 4) is underrepresented
    - BI-RADS 2,3,4 (Densities 1,2,3): 800 samples each
    - BI-RADS 5 (Density 4): 300 samples [UNDERREPRESENTED]
    
    4 classes: Labels 0, 1, 2, 3
    """
    
    def __init__(self, preprocess, location=os.path.expanduser('~/data'),
                 batch_size=128, num_workers=1):
        
        print("\n" + "="*70)
        print("BIRADS IMBALANCED DATASET (MULTI-CLASS)")
        print("="*70)
        print("Configuration:")
        print("  BI-RADS 2 (Density 1): 800 samples → Label 0")
        print("  BI-RADS 3 (Density 2): 800 samples → Label 1")
        print("  BI-RADS 4 (Density 3): 800 samples → Label 2")
        print("  BI-RADS 5 (Density 4): 300 samples → Label 3 [UNDERREPRESENTED]")
        
        # Load all data
        df = load_birads_data(location)
        
        # Sample according to imbalanced distribution
        sample_config = {
            1: 800,  # BI-RADS 2
            2: 800,  # BI-RADS 3
            3: 800,  # BI-RADS 4
            4: 300,  # BI-RADS 5 (UNDERREPRESENTED)
        }
        df_sampled = sample_distribution(df, sample_config, seed=42)
        
        # Train/test split
        train_data, test_data = self._split_data(df_sampled)
        
        # Create multi-class datasets
        self.train_dataset = MultiClassBIRADSDataset(train_data, preprocess)
        self.test_dataset = MultiClassBIRADSDataset(test_data, preprocess)
        
        # Create data loaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        # Multi-class classnames (labels 0-3)
        self.classnames = ['BI-RADS 2', 'BI-RADS 3', 'BI-RADS 4', 'BI-RADS 5']
        
        print(f"\nTrain: {len(train_data)} samples, Test: {len(test_data)} samples")
        print(f"Classnames: {self.classnames}")
        print("="*70 + "\n")
    
    def _split_data(self, df):
        """Split into train/test (80/20)"""
        data_list = df.to_dict('records')
        indices = list(range(len(data_list)))
        labels = [d['birads'] for d in data_list]
        
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=labels, random_state=42)
        
        train_data = [data_list[i] for i in train_idx]
        test_data = [data_list[i] for i in test_idx]
        
        train_dist = Counter([d['birads'] for d in train_data])
        test_dist = Counter([d['birads'] for d in test_data])
        
        print("\nTrain distribution:")
        for d in sorted(train_dist.keys()):
            print(f"  Density {d} (BI-RADS {d+1}): {train_dist[d]}")
        print("\nTest distribution:")
        for d in sorted(test_dist.keys()):
            print(f"  Density {d} (BI-RADS {d+1}): {test_dist[d]}")
        
        return train_data, test_data


# ============================================================================
# BINARY FOCUS DATASETS
# ============================================================================

class BinaryBIRADSDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for binary BI-RADS classification"""
    
    def __init__(self, data_list, preprocess, target_density):
        """
        Args:
            data_list: List of dicts with 'birads' and 'image_path'
            preprocess: CLIP preprocessing function
            target_density: Which density is the positive class (1 or 4)
        """
        self.data = data_list
        self.preprocess = preprocess
        self.target_density = target_density
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        try:
            image = Image.open(item['image_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading {item['image_path']}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply CLIP preprocessing
        if self.preprocess is not None:
            image = self.preprocess(image)
        
        # Binary label: 1 if target density, 0 otherwise
        label = 1 if item['birads'] == self.target_density else 0
        
        return image, label


class BIRADSDensity2Focus:
    """
    Binary focus on BI-RADS 2 (Density 1)
    - BI-RADS 2 (Density 1): 300 samples → Label 1 [FOCUS]
    - Not BI-RADS 2 (Densities 2,3,4): 300 samples (100 each) → Label 0
    
    2 classes: Labels 0, 1
    """
    
    def __init__(self, preprocess, location=os.path.expanduser('~/data'),
                 batch_size=128, num_workers=1):
        
        print("\n" + "="*70)
        print("BIRADS 2 FOCUS DATASET (BINARY)")
        print("="*70)
        print("Configuration:")
        print("  BI-RADS 2 (Density 1): 300 samples → Label 1 [FOCUS]")
        print("  Not BI-RADS 2:")
        print("    - BI-RADS 3 (Density 2): 100 samples")
        print("    - BI-RADS 4 (Density 3): 100 samples")
        print("    - BI-RADS 5 (Density 4): 100 samples")
        print("    Total Label 0: 300 samples")
        
        # Load all data
        df = load_birads_data(location)
        
        # Sample: 300 from Density 1, 100 each from others
        sample_config = {
            1: 300,  # BI-RADS 2 (FOCUS)
            2: 100,  # BI-RADS 3
            3: 100,  # BI-RADS 4
            4: 100,  # BI-RADS 5
        }
        df_sampled = sample_distribution(df, sample_config, seed=42)
        
        # Train/test split
        train_data, test_data = self._split_data(df_sampled, target_density=1)
        
        # Create binary datasets (target_density=1 means BI-RADS 2 is positive)
        self.train_dataset = BinaryBIRADSDataset(train_data, preprocess, target_density=1)
        self.test_dataset = BinaryBIRADSDataset(test_data, preprocess, target_density=1)
        
        # Create data loaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        # Binary classnames
        self.classnames = ['Not BI-RADS 2', 'BI-RADS 2']
        
        print(f"\nTrain: {len(train_data)} samples, Test: {len(test_data)} samples")
        print(f"Classnames: {self.classnames}")
        print("="*70 + "\n")
    
    def _split_data(self, df, target_density):
        """Split into train/test (80/20)"""
        data_list = df.to_dict('records')
        indices = list(range(len(data_list)))
        labels = [d['birads'] for d in data_list]
        
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=labels, random_state=42)
        
        train_data = [data_list[i] for i in train_idx]
        test_data = [data_list[i] for i in test_idx]
        
        # Count binary labels
        train_labels = [1 if d['birads'] == target_density else 0 for d in train_data]
        test_labels = [1 if d['birads'] == target_density else 0 for d in test_data]
        
        train_dist = Counter(train_labels)
        test_dist = Counter(test_labels)
        
        print("\nTrain distribution (binary):")
        print(f"  Label 0 (Not BI-RADS {target_density + 1}): {train_dist[0]}")
        print(f"  Label 1 (BI-RADS {target_density + 1}): {train_dist[1]}")
        print("\nTest distribution (binary):")
        print(f"  Label 0 (Not BI-RADS {target_density + 1}): {test_dist[0]}")
        print(f"  Label 1 (BI-RADS {target_density + 1}): {test_dist[1]}")
        
        return train_data, test_data


class BIRADSDensity5Focus:
    """
    Binary focus on BI-RADS 5 (Density 4)
    - BI-RADS 5 (Density 4): 300 samples → Label 1 [FOCUS]
    - Not BI-RADS 5 (Densities 1,2,3): 300 samples (100 each) → Label 0
    
    2 classes: Labels 0, 1
    """
    
    def __init__(self, preprocess, location=os.path.expanduser('~/data'),
                 batch_size=128, num_workers=1):
        
        print("\n" + "="*70)
        print("BIRADS 5 FOCUS DATASET (BINARY)")
        print("="*70)
        print("Configuration:")
        print("  BI-RADS 5 (Density 4): 300 samples → Label 1 [FOCUS]")
        print("  Not BI-RADS 5:")
        print("    - BI-RADS 2 (Density 1): 100 samples")
        print("    - BI-RADS 3 (Density 2): 100 samples")
        print("    - BI-RADS 4 (Density 3): 100 samples")
        print("    Total Label 0: 300 samples")
        
        # Load all data
        df = load_birads_data(location)
        
        # Sample: 100 each from Densities 1,2,3 and 300 from Density 4
        sample_config = {
            1: 100,  # BI-RADS 2
            2: 100,  # BI-RADS 3
            3: 100,  # BI-RADS 4
            4: 300,  # BI-RADS 5 (FOCUS)
        }
        df_sampled = sample_distribution(df, sample_config, seed=42)
        
        # Train/test split
        train_data, test_data = self._split_data(df_sampled, target_density=4)
        
        # Create binary datasets (target_density=4 means BI-RADS 5 is positive)
        self.train_dataset = BinaryBIRADSDataset(train_data, preprocess, target_density=4)
        self.test_dataset = BinaryBIRADSDataset(test_data, preprocess, target_density=4)
        
        # Create data loaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        # Binary classnames
        self.classnames = ['Not BI-RADS 5', 'BI-RADS 5']
        
        print(f"\nTrain: {len(train_data)} samples, Test: {len(test_data)} samples")
        print(f"Classnames: {self.classnames}")
        print("="*70 + "\n")
    
    def _split_data(self, df, target_density):
        """Split into train/test (80/20)"""
        data_list = df.to_dict('records')
        indices = list(range(len(data_list)))
        labels = [d['birads'] for d in data_list]
        
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=labels, random_state=42)
        
        train_data = [data_list[i] for i in train_idx]
        test_data = [data_list[i] for i in test_idx]
        
        # Count binary labels
        train_labels = [1 if d['birads'] == target_density else 0 for d in train_data]
        test_labels = [1 if d['birads'] == target_density else 0 for d in test_data]
        
        train_dist = Counter(train_labels)
        test_dist = Counter(test_labels)
        
        print("\nTrain distribution (binary):")
        print(f"  Label 0 (Not BI-RADS {target_density + 1}): {train_dist[0]}")
        print(f"  Label 1 (BI-RADS {target_density + 1}): {train_dist[1]}")
        print("\nTest distribution (binary):")
        print(f"  Label 0 (Not BI-RADS {target_density + 1}): {test_dist[0]}")
        print(f"  Label 1 (BI-RADS {target_density + 1}): {test_dist[1]}")
        
        return train_data, test_data
