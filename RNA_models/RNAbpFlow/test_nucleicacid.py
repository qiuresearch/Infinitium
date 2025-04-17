from src.data.nucleicacid import from_pdb_string, nucleicacid_to_model_features  

with open(path) as f:
    pdb_str = f.read()

rna = from_pdb_string(pdb_str)

features = nucleicacid_to_model_features(rna)
