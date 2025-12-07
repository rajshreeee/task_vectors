from ast import If
import os
from random import seed
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split


class BIRADS:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=128,
                 num_workers=1,
                 imbalanced=False,
                 max_samples_per_class=1000,
                 minority_samples=200,
                 seed=42, 
                focus_class=None):
        """
        BI-RADS dataset for mammography classification.
        
        Args:
            ...
            focus_class: If set, this density class gets minority_samples (usually more),
                        all others get max_samples_per_class (usually fewer)
        BI-RADS dataset for mammography classification.
        
        Dataset uses 0-indexed Density values: 0, 1, 2, 3, 4
        We use 1, 2, 3, 4 (skipping 0 which has only 4 samples)
        These map to labels 0, 1, 2, 3 for the model
        """
        
        self.location = Path(location)
        self.imbalanced = imbalanced
        self.seed = seed
        self.focus_class = focus_class  # ← ADD THIS

        # Load and prepare dataset
        data_dir = self.location / 'MINI-DDSM-Complete-JPEG-8'
        metadata_file = data_dir / 'DataWMask.xlsx'
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_file}")
        
        df_meta = pd.read_excel(metadata_file)
        
        # Map images to BI-RADS categories
        data = self._map_images_to_birads(data_dir, df_meta)
        df = pd.DataFrame(data)
        
        print(f"\n{'='*70}")
        print(f"BI-RADS Dataset Initialization")
        print(f"{'='*70}")
        print(f"Raw distribution (0-indexed Density values):")
        print(df['birads'].value_counts().sort_index())
        
        # Filter to Density 1-4 (skip 0 which has only 4 samples)
        # These are 0-indexed values from the dataset
        df = df[df['birads'].isin([1, 2, 3, 4])].copy()
        
        print(f"\nAfter filtering to Density 1-4:")
        print(df['birads'].value_counts().sort_index())
        
        # Apply imbalance if requested
        if imbalanced:
            df = self._create_imbalance(
                df, 
                max_samples_per_class=max_samples_per_class,
                minority_samples=minority_samples,
                seed=seed,
                focus_class=focus_class  # ← ADD THIS
            )
        
        # Create 0-indexed labels for model (shift down by 1)
        # BIRADS 2 ->  Density 1 -> label 0
        # BIRADS 3 -> Density 2 -> label 1
        # BIRADS 4 -> Density 3 -> label 2
        # BIRADS 5 -> Density 4 -> label 3
        df['label'] = df['birads']
        
        print(f"\nLabel mapping:")
        print(f"  Density -> Label")
        for density in sorted(df['birads'].unique()):
            label = density - 1
            count = len(df[df['birads'] == density])
            print(f"  {density} -> {label} ({count} samples)")
        
        # Convert to list of dicts to avoid Subset indexing issues
        data_list = df.to_dict('records')
        
        # Split indices for train/test
        indices = list(range(len(data_list)))
        labels_for_split = [d['birads'] for d in data_list]
        
        train_indices, test_indices = train_test_split(
            indices,
            test_size=0.2,
            stratify=labels_for_split,
            random_state=seed
        )
        
        train_data = [data_list[i] for i in train_indices]
        test_data = [data_list[i] for i in test_indices]
        
        print(f"\n{'='*70}")
        print(f"Dataset Split Summary")
        print(f"{'='*70}")
        print(f"Total samples: {len(data_list)}")
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")
        
        # Count distributions
        from collections import Counter
        train_dist = Counter([d['birads'] for d in train_data])
        test_dist = Counter([d['birads'] for d in test_data])
        
        print(f"\nTrain distribution (Density values):")
        for density in sorted(train_dist.keys()):
            print(f"  Density {density}: {train_dist[density]} samples")
        
        print(f"\nTest distribution (Density values):")
        for density in sorted(test_dist.keys()):
            print(f"  Density {density}: {test_dist[density]} samples")
        
        # Create datasets
        self.train_dataset = BIRADSDataset(train_data, preprocess)
        self.test_dataset = BIRADSDataset(test_data, preprocess)
        
        # Create data loaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        # Class names for zero-shot classification head
        # These correspond to Density 1, 2, 3, 4 (which are labels 0, 1, 2, 3)
        self.classnames = ['Density 1', 'Density 2', 'Density 3', 'Density 4']
        
        print(f"\nClassnames: {self.classnames}")
        print(f"Number of classes: {len(self.classnames)}")
        print(f"{'='*70}\n")
    
    def _create_imbalance(self, df, max_samples_per_class, minority_samples, seed, focus_class=None):
        """
        Create artificial imbalance with Density 4 as minority class
    
            Args:
            focus_class: If None, Density 4 is minority. 
                     If set, that density gets minority_samples (abundant),
                     all others get max_samples_per_class (scarce)
        """
        print(f"\n{'='*70}")
        print(f"Creating Artificial Imbalance")
        print(f"{'='*70}")
    
        if focus_class is None:
            # Default: Density 4 is minority (scarce)
            minority_class = 4
            majority_classes = [1, 2, 3]
            print(f"Config:")
            print(f"  Majority classes (Density 1-3): {max_samples_per_class} samples each")
            print(f"  Minority class (Density 4): {minority_samples} samples [SCARCE]")
        else:
            # Focus mode: specified class is abundant, others are scarce
            minority_class = focus_class
            majority_classes = [d for d in [1, 2, 3, 4] if d != focus_class]
            print(f"Config (FOCUS MODE):")
            print(f"  Regular classes (Densities {majority_classes}): {max_samples_per_class} samples each")
            print(f"  Focus class (Density {focus_class}): {minority_samples} samples [ABUNDANT]")
    
        print(f"  Imbalance ratio: {minority_samples / max_samples_per_class:.1f}:1")
    
    # Sample majority classes (scarce)
        dfs = []
        for density_val in majority_classes:
            df_class = df[df['birads'] == density_val]
            n_samples = min(len(df_class), max_samples_per_class)
            df_sampled = df_class.sample(n=n_samples, random_state=seed)
            dfs.append(df_sampled)
            status = "[SCARCE]" if focus_class is not None else ""
            print(f"  Density {density_val} {status}: {len(df_class)} -> {n_samples} samples")
    
    # Sample minority/focus class (abundant or scarce depending on mode)
        df_special = df[df['birads'] == minority_class]
        n_special = min(len(df_special), minority_samples)
        df_special_sampled = df_special.sample(n=n_special, random_state=seed)
        dfs.append(df_special_sampled)
    
        if focus_class is None:
            status = "[SCARCE MINORITY]"
        else:
            status = "[ABUNDANT FOCUS]"
        print(f"  Density {minority_class} {status}: {len(df_special)} -> {n_special} samples")
    
    # Combine and shuffle
        df_imbalanced = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    
        print(f"\nFinal distribution:")
        print(df_imbalanced['birads'].value_counts().sort_index())
        print(f"Total: {len(df_imbalanced)} samples")
    
        return df_imbalanced
    
    def _map_images_to_birads(self, data_dir, df_meta):
        """Map images to BI-RADS categories from metadata"""
        data = []
        
        class_folders = {
            'Normal': data_dir / 'Normal',
            'Benign': data_dir / 'Benign',
            'Cancer': data_dir / 'Cancer'
        }
        
        for class_name, class_path in class_folders.items():
            if not class_path.exists():
                print(f"Warning: {class_path} does not exist, skipping...")
                continue
            
            case_folders = [f for f in class_path.iterdir() if f.is_dir()]
            
            for case_folder in case_folders:
                case_id = case_folder.name
                
                # Find images (exclude masks)
                images = [img for img in case_folder.glob('*.jpg') 
                         if 'mask' not in img.name.lower() and 'boundary' not in img.name.lower()]
                
                for img_path in images:
                    # Match with metadata using fileName
                    matching_rows = df_meta[df_meta['fileName'].str.contains(case_id, na=False)]
                    
                    if len(matching_rows) == 0:
                        # Try fullPath match
                        matching_rows = df_meta[df_meta['fullPath'].str.contains(case_id, na=False)]
                    
                    if len(matching_rows) > 0:
                        row = matching_rows.iloc[0]
                        birads = int(row['Density'])  # Already 0-indexed: 0, 1, 2, 3, 4
                        
                        data.append({
                            'image_path': str(img_path),
                            'birads': birads,
                            'classification': class_name,
                            'age': row.get('Age', 50),
                            'status': row.get('Status', class_name)
                        })
        
        return data


class BIRADSDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for BI-RADS classification"""
    
    def __init__(self, data_list, preprocess):
        """
        Args:
            data_list: List of dictionaries with keys: 'image_path', 'label', 'birads', etc.
            preprocess: CLIP preprocessing function
        """
        self.data = data_list
        self.preprocess = preprocess
        
        # Verify all labels are valid (should be 0, 1, 2, 3)
        labels = [d['label'] for d in self.data]
        assert min(labels) >= 0, f"Found negative label: {min(labels)}"
        assert max(labels) <= 3, f"Found label > 3: {max(labels)} (expected 0-3)"
        
        print(f"  BIRADSDataset created with {len(self.data)} samples")
        print(f"  Label range: {min(labels)} to {max(labels)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        try:
            image = Image.open(item['image_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {item['image_path']}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply CLIP preprocessing
        if self.preprocess is not None:
            image = self.preprocess(image)
        
        label = item['label']
        
        # Safety check
        assert 0 <= label <= 3, f"Label {label} out of range [0, 3]"
        
        return image, label


# Convenience class for imbalanced version
class BIRADSImbalanced(BIRADS):
    """BI-RADS dataset with artificial imbalance (Density 4 is minority class)"""
    
    def __init__(self, preprocess, location=os.path.expanduser('~/data'), 
                 batch_size=128, num_workers=1):
        super().__init__(
            preprocess=preprocess,
            location=location,
            batch_size=batch_size,
            num_workers=num_workers,
            imbalanced=True,
            max_samples_per_class=1000,
            minority_samples=200,
            seed=42
        )

# Add these classes at the end of your birads.py file

class BIRADSDensity2Focus(BIRADS):
    """BI-RADS dataset with focus on Density 2 (BI-RADS 3) - make it abundant"""
    
    def __init__(self, preprocess, location=os.path.expanduser('~/data'), 
                 batch_size=128, num_workers=1):
        super().__init__(
            preprocess=preprocess,
            location=location,
            batch_size=batch_size,
            num_workers=num_workers,
            imbalanced=True,
            max_samples_per_class=300,   # Densities 1, 3, 4 get 300 each
            minority_samples=1200,        # Density 2 gets 1200 (4x more)
            seed=42,
            focus_class=2  # Focus on Density 2
        )


class BIRADSDensity4Focus(BIRADS):
    """BI-RADS dataset with focus on Density 4 (BI-RADS 5) - make it abundant"""
    
    def __init__(self, preprocess, location=os.path.expanduser('~/data'), 
                 batch_size=128, num_workers=1):
        super().__init__(
            preprocess=preprocess,
            location=location,
            batch_size=batch_size,
            num_workers=num_workers,
            imbalanced=True,
            max_samples_per_class=300,   # Densities 1, 2, 3 get 300 each
            minority_samples=1200,        # Density 4 gets 1200 (4x more)
            seed=42,
            focus_class=4  # Focus on Density 4
        )
