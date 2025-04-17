import os
from src.data.rnapair_dataset import RNAPairedDataset

def inspect_rnapair_dataset(folder_path, samples_per_seq=1):
    # Create dataset instance
    dataset = RNAPairedDataset(root_dir=folder_path, samples_per_seq=samples_per_seq)

    print(f"Total samples in dataset: {len(dataset)}\n")

    # Loop through and print one batch at a time
    for idx in range(len(dataset)):
        batch = dataset[idx]
        print(f"--- Sample {idx} ---")
        print(f"ID: {batch['id']}")
        print(f"Input Feats Keys: {list(batch['input_feats'].keys())}")
        print(f"Target Feats Keys: {list(batch['target_feats'].keys())}")
        
        # Example: print shapes
        print("Input Feats Shapes:")
        for k, v in batch['input_feats'].items():
            if hasattr(v, 'shape'):
                print(f"  {k}: {v.shape}")
            else:
                print(f"  {k}: {type(v)}")

        print("Target Feats Shapes:")
        for k, v in batch['target_feats'].items():
            if hasattr(v, 'shape'):
                print(f"  {k}: {v.shape}")
            else:
                print(f"  {k}: {type(v)}")

        print()

if __name__ == "__main__":
    folder = r"C:\Users\nikhi\Desktop\RNA\RNAbpFlow\complex\processed\rna3db-mmcifs\check_set\train"  # <-- Replace with your folder path
    inspect_rnapair_dataset(folder_path=folder, samples_per_seq=1)
