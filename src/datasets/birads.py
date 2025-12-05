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
                 num_workers=16):
        
        self.location = Path(location)
        
        # Load and prepare dataset
        data_dir = self.location / 'MINI-DDSM-Complete-JPEG-8'
        metadata_file = data_dir / 'DataWMask.xlsx'
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_file}")
        
        df_meta = pd.read_excel(metadata_file)
        
        # Map images to BI-RADS categories (1-4 only)
        data = self._map_images_to_birads(data_dir, df_meta)
        df = pd.DataFrame(data)
        
        # Filter to BI-RADS 1-5
        df = df[df['birads'].between(1, 4)]
        
        df['label'] = df['birads']
        
        # Train/test split (80/20, stratified)
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            stratify=df['birads'], 
            random_state=42
        )
        
        print(f"BI-RADS dataset loaded:")
        print(f"  Train samples: {len(train_df)}")
        print(f"  Test samples: {len(test_df)}")
        print(f"  Train distribution:\n{train_df['birads'].value_counts().sort_index()}")
        
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
        self.classnames = ['BI-RADS 2', 'BI-RADS 3', 'BI-RADS 4', 'BI-RADS 5']
    
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
