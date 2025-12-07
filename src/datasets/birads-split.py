import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle

def create_global_csv_split(location, test_samples_per_class=100, seed=42):
    """
    Create a single global train/test split and save as CSV files.
    
    Args:
        location: Path to data directory
        test_samples_per_class: Number of test samples per class (default 100)
        seed: Random seed for reproducibility
    """
    
    # Load data
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
    
    print(f"Total images loaded: {len(df)}")
    print("\nOriginal distribution:")
    print(df['birads'].value_counts().sort_index())
    
    # Split: 100 per class for test, rest for train
    train_dfs = []
    test_dfs = []
    
    print(f"\nCreating splits (test: {test_samples_per_class} per class, seed={seed}):")
    
    for birads_val in [1, 2, 3, 4]:
        df_class = df[df['birads'] == birads_val]
        n_available = len(df_class)
        n_test = min(test_samples_per_class, n_available)
        
        # Split this class
        df_train_class, df_test_class = train_test_split(
            df_class,
            test_size=n_test,
            random_state=seed
        )
        
        train_dfs.append(df_train_class)
        test_dfs.append(df_test_class)
        
        print(f"  Density {birads_val} (BI-RADS {birads_val + 1}):")
        print(f"    Train: {len(df_train_class)}, Test: {len(df_test_class)}")
    
    # Combine
    df_train = pd.concat(train_dfs, ignore_index=True)
    df_test = pd.concat(test_dfs, ignore_index=True)
    
    # Shuffle
    df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Save to CSV
    train_csv = data_dir / 'train_split.csv'
    test_csv = data_dir / 'test_split.csv'
    
    df_train.to_csv(train_csv, index=False)
    df_test.to_csv(test_csv, index=False)
    
    print(f"\n" + "="*70)
    print("SPLIT COMPLETE")
    print("="*70)
    print(f"Train CSV saved to: {train_csv}")
    print(f"  Total samples: {len(df_train)}")
    print(f"\nTest CSV saved to: {test_csv}")
    print(f"  Total samples: {len(df_test)}")
    
    print("\nTrain distribution:")
    print(df_train['birads'].value_counts().sort_index())
    
    print("\nTest distribution:")
    print(df_test['birads'].value_counts().sort_index())
    
    return train_csv, test_csv


if __name__ == '__main__':
    location = '/scratch/project_2016790/task_vectors/data'  # Adjust path
    train_csv, test_csv = create_global_csv_split(location, test_samples_per_class=100, seed=42)
    
    print("\n✓ Done! Use these CSV files in your datasets.")
