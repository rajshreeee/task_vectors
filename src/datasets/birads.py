import os
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
                 seed=42):
        """
        BI-RADS dataset for mammography classification.
        
        Args:
            preprocess: CLIP preprocessing function
            location: Root directory containing MINI-DDSM dataset
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            imbalanced: If True, create artificial imbalance (scarce BI-RADS 5)
            max_samples_per_class: Maximum samples for majority classes (BI-RADS 1-4)
            minority_samples: Number of samples for minority class (BI-RADS 5)
            seed: Random seed for reproducibility
        """
        
        self.location = Path(location)
        self.imbalanced = imbalanced
        self.seed = seed
        
        # Load and prepare dataset
        data_dir = self.location / 'MINI-DDSM-Complete-JPEG-8'
        metadata_file = data_dir / 'DataWMask.xlsx'
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_file}")
        
        df_meta = pd.read_excel(metadata_file)
        
        # Map images to BI-RADS categories
        data = self._map_images_to_birads(data_dir, df_meta)
        df = pd.DataFrame(data)
        
        # Filter to BI-RADS 1-5 (Density column values 1-4, where 4 maps to BI-RADS 5)
        # Based on your data: Density values are 0,1,2,3,4
        # We use 1,2,3,4 which correspond to BI-RADS 1,2,3,4,5 in medical terms
        df = df[df['birads'].between(1, 4)]
        
        print(f"\n{'='*70}")
        print(f"BI-RADS Dataset Initialization")
        print(f"{'='*70}")
        print(f"Original distribution (Density column 1-4):")
        print(df['birads'].value_counts().sort_index())
        
        # Apply imbalance if requested
        if imbalanced:
            df = self._create_imbalance(
                df, 
                max_samples_per_class=max_samples_per_class,
                minority_samples=minority_samples,
                seed=seed
            )
        df['label'] = df['birads']

        # Train/test split (80/20, stratified)
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            stratify=df['birads'], 
            random_state=seed
        )
        
        print(f"\n{'='*70}")
        print(f"Dataset Split Summary")
        print(f"{'='*70}")
        print(f"Total samples: {len(df)}")
        print(f"Train samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        print(f"\nTrain distribution:")
        print(train_df['birads'].value_counts().sort_index())
        print(f"\nTest distribution:")
        print(test_df['birads'].value_counts().sort_index())
        
        # Create datasets
        self.train_dataset = BIRADSDataset(train_df, preprocess)
        self.test_dataset = BIRADSDataset(test_df, preprocess)
        
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
        # Map to actual BI-RADS categories
        self.classnames = ['BI-RADS 1', 'BI-RADS 2', 'BI-RADS 3', 'BI-RADS 4']
        
        print(f"\nClassnames: {self.classnames}")
        print(f"{'='*70}\n")
    
    def _create_imbalance(self, df, max_samples_per_class, minority_samples, seed):
        """Create artificial imbalance with BI-RADS 4 (Density=4) as minority class"""
        
        print(f"\n{'='*70}")
        print(f"Creating Artificial Imbalance")
        print(f"{'='*70}")
        print(f"Config:")
        print(f"  Majority classes (BI-RADS 1-3): {max_samples_per_class} samples each")
        print(f"  Minority class (BI-RADS 4): {minority_samples} samples")
        print(f"  Imbalance ratio: {max_samples_per_class / minority_samples:.1f}:1")
        
        # Downsample majority classes (Density 1, 2, 3)
        dfs = []
        for birads_val in [1, 2, 3]:
            df_class = df[df['birads'] == birads_val]
            n_samples = min(len(df_class), max_samples_per_class)
            df_sampled = df_class.sample(n=n_samples, random_state=seed)
            dfs.append(df_sampled)
            print(f"  BI-RADS {birads_val} (Density={birads_val}): {len(df_class)} -> {n_samples} samples")
        
        # Make BI-RADS 4/5 (Density 4) the minority class
        df_minority = df[df['birads'] == 4]
        n_minority = min(len(df_minority), minority_samples)
        df_minority_sampled = df_minority.sample(n=n_minority, random_state=seed)
        dfs.append(df_minority_sampled)
        print(f"  BI-RADS 4 (Density=4) [MINORITY]: {len(df_minority)} -> {n_minority} samples")
        
        # Combine and shuffle
        df_imbalanced = pd.concat(dfs).sample(frac=1, random_state=seed).reset_index(drop=True)
        
        print(f"\nFinal imbalanced distribution:")
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
                        birads = int(row['Density'])
                        
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
    
    def __init__(self, df, preprocess):
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        try:
            image = Image.open(row['image_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {row['image_path']}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply CLIP preprocessing
        if self.preprocess is not None:
            image = self.preprocess(image)
        
        label = row['label']
        
        return image, label


# Convenience class for imbalanced version
class BIRADSImbalanced(BIRADS):
    """BI-RADS dataset with artificial imbalance (BI-RADS 4 is minority class)"""
    
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
