import os
import torch
from torch.utils.data import Dataset
from src.data.nucleicacid import from_pdb_string, nucleicacid_to_model_features
from src.data.data_transform import make_atom_mask
import numpy as np
class RNAPairedDataset(Dataset):
    def __init__(self, root_dir, samples_per_seq=1, seed_start=42):
        self.root_dir = root_dir
        self.samples_per_seq = samples_per_seq
        self.seed_start = seed_start
        self.sample_list = []

        self.sample_ids = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        for pdb_index, pdb_id in enumerate(self.sample_ids):
            ss_path = os.path.join(root_dir, pdb_id)
            npy_files = [f for f in os.listdir(ss_path) if f.endswith('.npy')]
            npy_files = sorted(npy_files)[:3]

            ss_maps = [np.load(os.path.join(ss_path, f)) for f in npy_files]
            seqlen = ss_maps[0].shape[0]

            if len(ss_maps) == 1:
                ss_final = np.stack([ss_maps[0]] * 3, axis=-1)
            elif len(ss_maps) == 2:
                ss_final = np.stack([ss_maps[0], ss_maps[1], ss_maps[0]], axis=-1)
            else:
                ss_final = np.stack(ss_maps, axis=-1)

            for sample_id in range(samples_per_seq):
                seed = seed_start * (sample_id + 1) * 10  
                self.sample_list.append((pdb_id, ss_final, seed))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_id, ss, seed = self.sample_list[idx]
        sample_dir = os.path.join(self.root_dir, sample_id)

        with open(os.path.join(sample_dir, f"{sample_id}.pdb"), 'r') as f:
            pdb_str = f.read()

        na_obj = from_pdb_string(pdb_str)
        target_feats = nucleicacid_to_model_features(na_obj)

        input_feats = make_atom_mask(sample_id, self.root_dir)
        input_feats["ss"] = ss
        input_feats["seed"] = seed

        return {
            "target_feats": target_feats,
            "input_feats": input_feats,
            "id": sample_id
        }
