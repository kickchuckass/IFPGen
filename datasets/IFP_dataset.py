import os
import pickle
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import Dataset, Subset
from tqdm.auto import tqdm

# Suppress noisy loggers
logging.getLogger('plip').setLevel(logging.ERROR)
logging.getLogger('unimol_tools').setLevel(logging.ERROR)

# Add parent directory to path for custom imports
import sys
sys.path.append('..')

from datasets.pl_data import ProteinLigandData, torchify_dict
from datasets.plip_2 import IFPAnalyzer, ALLOW_HET_LIST
from unimol_tools import UniMolRepr
from utils.data import parse_sdf_file, segment_and_mask_brics
from utils.protein import PDBProtein


# Global constants
INTERACTION_TYPES = ["pipi", "anion", "cation", "hbd", "hba", "hydro", "metal", "halogen"]
ELSE = None
LIG_ATOM_SYMBOLS = [
    ELSE, "C", "N", "O", "F", "S", "P", "Cl", "Br", "I"
]

# Initialize UniMol for ligand representation
unimol_repr = UniMolRepr(
    data_type='molecule',
    remove_hs=True,
    model_name='unimolv1',
    model_size='84m'
)


def sanitize_pocket_interaction(inter_dict: dict, ligand_n: int) -> dict:
    """
    Adjust receptor atom indices in interaction dictionary by subtracting ligand atom count.

    Args:
        inter_dict: Dictionary of interactions with [receptor_idx, ligand_idx] pairs.
        ligand_n: Number of ligand atoms.

    Returns:
        Updated interaction dictionary with corrected receptor indices.
    """
    return {k: [[p - ligand_n, l] for p, l in v] for k, v in inter_dict.items()}


def filter_ligand(mol: Chem.Mol, num_atom_min_max: Tuple[int, int] = (6, 60),
                  avail_atom_types: List[Optional[str]] = LIG_ATOM_SYMBOLS) -> bool:
    """
    Filter ligand molecule based on size and atom types.

    Args:
        mol: RDKit molecule.
        num_atom_min_max: Min/max number of atoms allowed.
        avail_atom_types: List of allowed atom symbols.

    Returns:
        True if ligand passes filters.
    """
    if mol is None or len(mol.GetConformers()) == 0:
        return False
    num_atoms = mol.GetNumAtoms()
    if not (num_atom_min_max[0] <= num_atoms <= num_atom_min_max[1]):
        return False
    return all(atom.GetSymbol() in avail_atom_types for atom in mol.GetAtoms())


def filter_receptor(mol: Chem.Mol, num_atom_min_max: Tuple[int, int] = (140, 660)) -> bool:
    """
    Filter receptor molecule based on size.

    Args:
        mol: RDKit molecule.
        num_atom_min_max: Min/max number of atoms allowed.

    Returns:
        True if receptor passes filter.
    """
    if mol is None or len(mol.GetConformers()) == 0:
        return False
    num_atoms = mol.GetNumAtoms()
    return num_atom_min_max[0] <= num_atoms <= num_atom_min_max[1]


def get_dataset(config, **kwargs):
    """
    Create dataset and optionally split into subsets.

    Args:
        config: Configuration object.
        **kwargs: Additional arguments for IfpDiffDataset.

    Returns:
        Dataset or (dataset, subsets) if split is provided.
    """
    dataset = IfpDiffDataset(**kwargs)
    if hasattr(config, 'split') and config.split:
        split = torch.load(config.split)
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    return dataset


class IfpDiffDataset(Dataset):
    """
    Dataset for IFP-conditioned diffusion model training.
    Processes protein-ligand complexes with IFP, UniMol, and BRICS fragmentation.
    """

    def __init__(
        self,
        transform=None,
        overwrite: bool = False,
        kekulize: bool = True,
        skip_exit: bool = True,
        if_H: bool = False,
        if_metal: bool = True,
        index: str = '/home/worker/users/WYG/IFP-diffusion_v2/datasets/250327_dataset_process_B_M/index_combine_index_C_P_M.pkl',
        processed_path: str = '/home/worker/users/WYG/IFP-diffusion_v2/datasets/combination_processed_v4'
    ):
        super().__init__()
        self.transform = transform
        self.kekulize = kekulize
        self.overwrite = overwrite
        self.skip_exit = skip_exit
        self.if_H = if_H
        self.if_metal = if_metal
        self.processed_path = processed_path

        # Load index
        self.index = pickle.load(open(index, 'rb'))

        # Process data if needed
        if not os.path.exists(self.processed_path) or self.overwrite:
            print(f"{self.processed_path} does not exist or overwrite=True. Starting processing...")
            self._process()

        print(f"Dataset loaded from: {self.processed_path}")

    def _process(self):
        """Process all entries in the index and save to processed_path."""
        os.makedirs(self.processed_path, exist_ok=True)
        num_skipped = 0
        skipped_files = []
        valid_count = 0

        for orig_idx, item in tqdm(enumerate(self.index), desc="Processing", mininterval=10):
            output_file = os.path.join(self.processed_path, f"{valid_count}_processed.pkl")

            # Skip if already processed and skip_exit is True
            if os.path.exists(output_file) and self.skip_exit:
                valid_count += 1
                continue

            try:
                # Parse input paths
                pocket_nometal_path, ligand_path = item[:2]
                pocket_path = item[2] if len(item) > 2 else None

                # Load protein
                protein = PDBProtein(
                    pocket_path or pocket_nometal_path,
                    allow_het=ALLOW_HET_LIST if self.if_metal else [],
                    if_H=self.if_H
                )
                pocket_dict = protein.to_dict_atom()

                # Load ligand
                ligand_dict = parse_sdf_file(ligand_path)
                data = ProteinLigandData.from_protein_ligand_dicts(
                    protein_dict=torchify_dict(pocket_dict),
                    ligand_dict=torchify_dict(ligand_dict)
                ).to_dict()

                # Add metadata
                data.update({
                    'protein_filename': pocket_path or pocket_nometal_path,
                    'ligand_filename': ligand_path
                })

                # Compute IFP
                analyzer = IFPAnalyzer(if_H=self.if_H, if_metal=self.if_metal)
                ifp_tensor, _ = analyzer.GET_IFP(ligand_path, pocket_nometal_path, pocket_path)
                data['ifp_info'] = ifp_tensor

                # UniMol representation
                unimol_vec = unimol_repr.get_repr(ligand_path, return_atomic_reprs=False)['cls_repr'][0]
                data['unimol_repr'] = torch.from_numpy(unimol_vec).reshape(1, 512)

                # BRICS fragmentation mask for inpainting
                rdmol = Chem.MolFromMolFile(ligand_path, sanitize=False)
                Chem.SanitizeMol(rdmol)
                rdmol = Chem.RemoveHs(rdmol)
                data['inpainting_mask'] = torch.tensor(segment_and_mask_brics(rdmol), dtype=torch.bool)

                # Save processed data
                with open(output_file, 'wb') as f:
                    pickle.dump(data, f)

                valid_count += 1

            except Exception as e:
                num_skipped += 1
                error_msg = f"Failed to process {ligand_path}: {str(e)}\n"
                skipped_files.append(f"[{orig_idx}] {error_msg}")
                print(error_msg.strip())

        # Save skip log
        skip_log_path = os.path.join(self.processed_path, 'skipped_log.txt')
        with open(skip_log_path, 'w') as f:
            f.writelines(skipped_files)
        print(f"Processing complete. Skipped: {num_skipped}, Saved: {valid_count}")
        print(f"Skip log saved to: {skip_log_path}")

    def __len__(self) -> int:
        return len([f for f in os.listdir(self.processed_path) if f.endswith('_processed.pkl')])

    def __getitem__(self, idx: int) -> ProteinLigandData:
        file_path = os.path.join(self.processed_path, f"{idx}_processed.pkl")
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
        data = ProteinLigandData(**data_dict)
        if self.transform:
            data = self.transform(data)
        return data


if __name__ == '__main__':
    print("START")
    processed_path = '/home/worker/users/WYG/IFP-diffusion_v2/datasets/combination_processed_v4_T_H_refined_inpainting'
    index_path = '/home/worker/users/WYG/IFP-diffusion_v2/datasets/250603_New_model_test/CPM+T_(withM)/index_combine_index_C_P_M_T.pkl'
    
    dataset = IfpDiffDataset(
        overwrite=True,
        skip_exit=True,
        index=index_path,
        processed_path=processed_path,
        if_H=True,
        if_metal=True
    )
    print(f"END - Dataset size: {len(dataset)}")