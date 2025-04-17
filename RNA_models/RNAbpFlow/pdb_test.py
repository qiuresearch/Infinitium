import io
import numpy as np
from typing import Optional, Dict, List
from Bio.PDB import PDBParser

# Base constants from your base_constants.py
restype_1to3 = {
    'A': 'A',
    'G': 'G',
    'C': 'C',
    'U': 'U',
}
restype_3to1 = {v: k for k, v in restype_1to3.items()}

# Define a restype name for all unknown residues.
unk_restype = 'UNK'
restypes = ['A', 'G', 'C', 'U']
resnames = [restype_1to3[r] for r in restypes] + [unk_restype]

resname_to_idx = {resname: i for i, resname in enumerate(resnames)}

# Additional constants needed for the parser
# Common nucleic acid atoms
atom_types = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", 
              "C2'", "O2'", "C1'", "N9", "C8", "N7", "C5", "C6", "N6", 
              "N1", "C2", "N3", "C4", "O6", "N4", "N2", "O2", "O4"]

# Create atom order mapping
atom_order = {atom: i for i, atom in enumerate(atom_types)}
atom_type_num = len(atom_types)

# Create restype order with proper indexing
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes) + 1  # Include UNK

class NucleicAcid:
    def __init__(self, atom_positions, atom_mask, aatype, residue_index, b_factors, chain_index=None):
        self.atom_positions = atom_positions
        self.atom_mask = atom_mask
        self.aatype = aatype
        self.residue_index = residue_index
        self.b_factors = b_factors
        self.chain_index = chain_index

def from_pdb_string(
    pdb_str: str,
    chain_id: Optional[str] = None,
    print_nucleotides: bool = True
) -> NucleicAcid:
    """Takes a PDB string and constructs a NucleicAcid object.

    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.

    Args:
        pdb_str: The contents of the pdb file
        chain_id: If None, then the pdb file must contain a single chain (which
        will be parsed). If chain_id is specified (e.g. A), then only that chain
        is parsed.
        print_nucleotides: If True, print information about each nucleotide found

    Returns:
        A new `NucleicAcid` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('none', pdb_fh)
   
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f'Only single model PDBs are supported. Found {len(models)} models.'
        )
    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    # For printing nucleotide information
    nucleotide_info = []
    
    chains_found = False
    
    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
            
        chains_found = True
        nucleotide_count = 0
        
        if print_nucleotides:
            print(f"Processing chain: {chain.id}")
            
        for res in chain:
            if res.id[2] != ' ':
                raise ValueError(
                    f'PDB contains an insertion code at chain {chain.id} and residue '
                    f'index {res.id[1]}. These are not supported.'
                )
            # Inside the loop over residues:
            
            # Get residue name and convert to one-letter code if possible
            res_name = res.resname.strip()
            res_shortname = restype_3to1.get(res_name, 'X')
            
            # Determine residue type index
            if res_shortname in restype_order:
                restype_idx = restype_order[res_shortname]
            else:
                # Use the last index for unknown residues
                restype_idx = restype_num - 1
            
            pos = np.zeros((atom_type_num, 3))
            mask = np.zeros((atom_type_num,))
            res_b_factors = np.zeros((atom_type_num,))
            # Inside the loop over residues:
            print(f"Debug - Residue name: '{res_name}', Looking up in restype_3to1: {res_name in restype_3to1}")
            # Track atoms found in this residue
            atoms_found = []
            
            for atom in res:
                if atom.name not in atom_types:
                    continue
                pos[atom_order[atom.name]] = atom.coord
                mask[atom_order[atom.name]] = 1.
                res_b_factors[atom_order[atom.name]] = atom.bfactor
                atoms_found.append(atom.name)

            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
                
            # Store residue info for printing
            nucleotide_count += 1
            if print_nucleotides:
                # Calculate center
                valid_positions = pos[mask > 0.5]
                if len(valid_positions) > 0:
                    center = np.mean(valid_positions, axis=0)
                else:
                    center = np.array([0.0, 0.0, 0.0])
                    
                nucleotide_info.append({
                    'chain': chain.id,
                    'residue_name': res_name,
                    'short_name': res_shortname,
                    'position': res.id[1],
                    'num_atoms': len(atoms_found),
                    'atoms': atoms_found,
                    'center': center
                })
                
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)
    
    if not chains_found:
        if chain_id:
            raise ValueError(f"No chain with ID '{chain_id}' found in the PDB file")
        else:
            raise ValueError("No chains found in the PDB file")
    
    # Print nucleotide information if requested
    if print_nucleotides and nucleotide_info:
        print("\nNucleotides found in the PDB:")
        print("-" * 80)
        print(f"{'Chain':<6}{'Residue':<8}{'Type':<6}{'Position':<10}{'Atoms':<8}{'Center':<30}")
        print("-" * 80)
        
        for nuc in nucleotide_info:
            print(f"{nuc['chain']:<6}{nuc['residue_name']:<8}{nuc['short_name']:<6}{nuc['position']:<10}"
                  f"{nuc['num_atoms']:<8}({nuc['center'][0]:6.2f}, {nuc['center'][1]:6.2f}, {nuc['center'][2]:6.2f})")
        
        print("-" * 80)
        print(f"Total nucleotides found: {len(nucleotide_info)}")
    
    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return NucleicAcid(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        b_factors=np.array(b_factors),
        chain_index=chain_index
    )

# Example usage:
def read_pdb_file(file_path, chain_id=None):
    """Read a PDB file and parse it into a NucleicAcid object"""
    with open(file_path, 'r') as f:
        pdb_str = f.read()
    return from_pdb_string(pdb_str, chain_id)

# If this module is run directly, demonstrate usage with a sample file
if __name__ == "__main__":
    import sys
    
    pdb_file=r"C:\Users\nikhi\Desktop\RNA\RNAbpFlow\complex\processed\rna3db-mmcifs\check_set\component_30\3sd3_A\3sd3_A.pdb"
    chain = None    
    try:
        with open(pdb_file, 'r') as f:
            pdb_content = f.read()
        
        nucleic_acid = from_pdb_string(pdb_content, chain)
        print(f"\nParsed nucleic acid structure:")
        print(f"  - Number of residues: {len(nucleic_acid.aatype)}")
        print(f"  - Residue types: {np.unique(nucleic_acid.aatype)}")
        
    except FileNotFoundError:
        print(f"Error: PDB file '{pdb_file}' not found.")
    except Exception as e:
        print(f"Error: {str(e)}")
    