import os
import sys
import argparse
import pickle
import shutil
import itertools
from collections import namedtuple
from functools import partial
import multiprocessing as mp
import numpy as np
import torch
import pandas as pd
import tempfile
from plip.structure.preparation import Mol, PDBComplex
from rdkit import Chem
from typing import List, Dict, Set, Optional

# Add parent directory to path for local module imports
sys.path.append('..')
from utils.data import parse_sdf_file
from utils.protein import PDBProtein
from typing import List, Dict, Set, Optional, Tuple

# Global constants
ALLOW_HET_LIST = ["NA", "K", "MG", "CA", "FE", "CU", "ZN", "MN", "NI", "CO"]
METAL_ELEMENT = [11, 19, 12, 20, 26, 29, 30, 25, 28, 27]
METAL_DIST_MAX = 3.0
METAL_BINDING_TYPES = [8, 7, 16, 15, 9, 17, 35, 53]
INTERACTION_TYPES = ["pipi", "anion", "cation", "hbd", "hba", "hydro", "metal", "halogen"]
AA_NAME_SYM = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V', 'HET': 'X'
}
AA_NUMBER_NAME = {i: k for i, (k, _) in enumerate(AA_NAME_SYM.items())}
INTERACTION_COLUMNS = [
    "salt_bridge_anion",
    "salt_bridge_cation",
    "hydrogen_bond_donor",
    "hydrogen_bond_acceptor",
    "hydrophobic_interaction",
    "pi_stacking",
    "metal",
    "halogen bonds"
]


def get_ifp_score(ref_ifp: torch.Tensor, query_ifp: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute cosine similarity between reference and query IFP vectors, per residue.
    Only consider residues with at least one interaction in the reference.
    """
    similarities = []
    num_valid = 0
    for p_vec, g_vec in zip(ref_ifp.float(), query_ifp.float()):
        mask = (p_vec > 0)
        if mask.sum() == 0:
            continue
        p_required = p_vec[mask]
        g_required = g_vec[mask]
        dot_prod = torch.dot(p_required, g_required)
        norm_p = torch.norm(p_required)
        norm_g = torch.norm(g_required)
        sim = dot_prod / (norm_p * norm_g + eps)
        similarities.append(sim)
        num_valid += 1

    # Return average similarity; if no valid residues, return 1.0
    return 1.0 if num_valid == 0 else sum(similarities) / num_valid


class IFPAnalyzer:
    def __init__(
        self,
        allow_het_list: List[str] = ALLOW_HET_LIST,
        metal_element: List[int] = METAL_ELEMENT,
        metal_dist_max: float = METAL_DIST_MAX,
        metal_binding_types: List[int] = METAL_BINDING_TYPES,
        if_H: bool = False,
        if_metal: bool = True
    ):
        """
        Initialize IFP analyzer with metal and H-atom handling options.
        """
        self.allow_het_list = allow_het_list if if_metal else []
        self.metal_element = metal_element
        self.metal_dist_max = metal_dist_max
        self.metal_binding_types = metal_binding_types
        self.if_H = if_H  # Whether to include hydrogen atoms in analysis

    # -------------------------------
    # Utility Methods
    # -------------------------------
    @staticmethod
    def euclidean3d(coord1: np.ndarray, coord2: np.ndarray) -> float:
        """Compute Euclidean distance between two 3D coordinates."""
        return np.linalg.norm(np.array(coord1) - np.array(coord2))

    @staticmethod
    def vecangle(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute angle (in degrees) between two vectors."""
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    # -------------------------------
    # Core Methods
    # -------------------------------
    def join_complex(
        self,
        ligand_fn: str,
        pocket_fn: str,
        complex_fn: Optional[str] = None,
        LIG_USE_AROMATIC: bool = True
    ) -> Tuple[Chem.Mol, str]:
        """
        Merge ligand and pocket into a single PDB complex.
        Forces ligand residue name to 'LIG'.
        """
        os.makedirs('../tmp/', exist_ok=True)
        if complex_fn is None:
            fd, complex_fn = tempfile.mkstemp(
                suffix=".pdb",
                prefix="temp_process_crossdocked_plip_input_",
                dir='../tmp/'
            )
            os.close(fd)

        lig_mol = Chem.SDMolSupplier(ligand_fn, sanitize=False)[0]
        try:
            lig_mol = Chem.RemoveHs(lig_mol)
        except Exception:
            pass

        if not LIG_USE_AROMATIC:
            Chem.Kekulize(lig_mol, clearAromaticFlags=True)
        else:
            Chem.rdmolops.SanitizeMol(
                lig_mol,
                sanitizeOps=Chem.rdmolops.SanitizeFlags.SANITIZE_SETAROMATICITY
            )
        num_ligand_atom = lig_mol.GetNumAtoms()

        command = f"obabel {ligand_fn} {pocket_fn} -O {complex_fn} -j -d 2> /dev/null"
        os.system(command)

        with open(complex_fn, "r") as f:
            lines = f.readlines()
        new_lines = []
        for i, line in enumerate(lines):
            if 1 < i < num_ligand_atom + 2:
                new_line = line[:17] + "LIG" + line[20:25] + "1 " + line[27:]
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        with open(complex_fn, "w") as f:
            f.writelines(new_lines)

        complex_mol = Chem.MolFromPDBFile(complex_fn, sanitize=False)
        return complex_mol, complex_fn

    def get_complex_interaction_info(self, complex_fn: str) -> Optional[Dict]:
        """
        Use PLIP to analyze protein-ligand interactions in the complex.
        Returns a dictionary with interaction atom indices.
        """
        metal_list = ["NA", "K", "MG", "CA", "FE", "CU", "ZN", "MN", "NI", "CO"]
        out = {}
        output_path = '../../tmp/'
        my_mol = PDBComplex(output_path=output_path)
        my_mol.load_pdb(complex_fn)
        ligs = [
            ":".join([x.hetid, x.chain, str(x.position)])
            for x in my_mol.ligands
            if x.hetid == "LIG" or x.hetid not in metal_list
        ]
        if len(ligs) == 0:
            return None

        my_mol.analyze()
        my_interactions = my_mol.interaction_sets[ligs[0]]

        anions = my_interactions.saltbridge_pneg
        cations = my_interactions.saltbridge_lneg
        hbds = my_interactions.hbonds_pdon
        hbas = my_interactions.hbonds_ldon
        hydros = my_interactions.hydrophobic_contacts
        pipis = my_interactions.pistacking
        metal = my_interactions.metal_complexes
        halogen_bonds = my_interactions.halogen_bonds

        # Extract atom indices for each interaction type
        anions_idx = [x - 1 for x in itertools.chain(*[x.negative.atoms_orig_idx for x in anions])]
        out["SBA"] = list(set(anions_idx))

        cations_idx = [x - 1 for x in itertools.chain(*[x.positive.atoms_orig_idx for x in cations])]
        out["SBC"] = list(set(cations_idx))

        out["HBD"] = list(set(x.d_orig_idx - 1 for x in hbds))
        out["HBA"] = list(set(x.a_orig_idx - 1 for x in hbas))
        out["HI"] = list(set(hyd.bsatom_orig_idx - 1 for hyd in hydros))
        out["PP"] = list(set(x - 1 for x in itertools.chain(*[x.proteinring.atoms_orig_idx for x in pipis])))
        out["MET"] = list(set(x.metal_orig_idx - 1 for x in metal))
        out['halogen_bonds'] = list(set(x.don_orig_idx - 1 for x in halogen_bonds))

        return out

    def determine_coordination_geometry(
        self,
        metal: dict,
        targets: list
    ) -> Tuple[bool, str, int]:
        """
        Determine metal coordination geometry based on target atom positions.
        Returns: (has_ligand, geometry, coordination_number)
        """
        num_targets = len(targets)
        if num_targets == 0:
            return False, 'NA', 0

        ideal_angles = {
            2: {'linear': [180.0]},
            3: {'trigonal.planar': [120.0, 120.0]},
            4: {
                'tetrahedral': [109.5, 109.5, 109.5],
                'square.planar': [90.0, 90.0, 90.0, 90.0]
            },
            5: {
                'trigonal.bipyramidal': [90.0, 120.0, 120.0],
                'square.pyramidal': [90.0, 90.0, 90.0]
            },
            6: {'octahedral': [90.0, 90.0, 90.0, 90.0, 180.0]}
        }

        vectors = [pos - metal['pos'] for _, pos, _ in targets]
        angles = []
        if_ligand = []
        for i, j in itertools.combinations(range(num_targets), 2):
            angle = self.vecangle(vectors[i], vectors[j])
            angles.append(angle)
            if targets[i][2] == 'ligand' or targets[j][2] == 'ligand':
                if_ligand.append(True)
            else:
                if_ligand.append(False)

        angles = np.array(angles)
        if_ligand = np.array(if_ligand)
        sorted_idx = np.argsort(angles)
        sorted_angles = angles[sorted_idx]
        sorted_if_ligand = if_ligand[sorted_idx]

        best_geometry = 'NA'
        best_coordination_num = num_targets
        min_rms = float('inf')
        best_num_angles = len(sorted_angles)

        for coo in range(2, min(7, num_targets + 1)):
            if coo in ideal_angles:
                for geometry, ideal in ideal_angles[coo].items():
                    n_angles = len(ideal)
                    current_angles = sorted_angles[:n_angles]
                    sorted_ideal = np.sort(ideal)
                    rms = np.sqrt(np.mean((current_angles - sorted_ideal) ** 2))
                    if rms < min_rms:
                        min_rms = rms
                        best_geometry = geometry
                        best_coordination_num = coo
                        best_num_angles = n_angles

        has_ligand = bool(np.any(sorted_if_ligand[:best_num_angles]))
        return has_ligand, best_geometry, best_coordination_num

    def _get_pocket_interaction_matrix(
        self,
        ligand_n: int,
        pocket_n: int,
        info: Dict
    ) -> np.ndarray:
        """
        Build interaction matrix for pocket atoms based on IFP info.
        """
        anion = info.get('SBA', [])
        cation = info.get('SBC', [])
        hbd = info.get('HBD', [])
        hba = info.get('HBA', [])
        hydro = info.get('HI', [])
        pipi = info.get('PP', [])
        metal = info.get('MET', [])
        halogen = info.get('halogen_bonds', [])

        pocket_intr_vectors = []
        for i in range(pocket_n):
            vec = [0] * len(INTERACTION_TYPES)
            if i + ligand_n in anion:
                vec[0] = 1
            if i + ligand_n in cation:
                vec[1] = 1
            if i + ligand_n in hbd:
                vec[2] = 1
            if i + ligand_n in hba:
                vec[3] = 1
            if i + ligand_n in hydro:
                vec[4] = 1
            if i + ligand_n in pipi:
                vec[5] = 1
            if i + ligand_n in metal:
                vec[6] = 1
            if i + ligand_n in halogen:
                vec[7] = 1
            pocket_intr_vectors.append(vec)

        return np.stack(pocket_intr_vectors, axis=0)

    def analyze_ligand_metal_coordination(
        self,
        metal: dict,
        ligand_atoms: dict,
        protein_atoms: Optional[dict] = None
    ) -> dict:
        """
        Analyze if ligand (and protein) coordinates with metal and determine geometry.
        """
        metal_binding_lig = [
            i for i, atom in enumerate(ligand_atoms['element'])
            if atom in self.metal_binding_types
        ]
        metal_binding_protein = [
            i for i, atom in enumerate(protein_atoms['element'])
            if atom in self.metal_binding_types
        ] if protein_atoms else []

        targets = []
        for idx in metal_binding_lig:
            distance = self.euclidean3d(metal['pos'], ligand_atoms['pos'][idx])
            if distance < self.metal_dist_max:
                targets.append((idx, ligand_atoms['pos'][idx], 'ligand'))

        if not targets:
            return {
                'is_coordinated': False,
                'coordinated_atoms': [],
                'geometry': 'NA',
                'coordination_num': 0
            }

        for idx in metal_binding_protein:
            distance = self.euclidean3d(metal['pos'], protein_atoms['pos'][idx])
            if distance < self.metal_dist_max:
                targets.append((idx, protein_atoms['pos'][idx], 'protein'))

        has_ligand, geometry, coordination_num = self.determine_coordination_geometry(metal, targets)
        return {
            'is_coordinated': has_ligand,
            'geometry': geometry,
            'coordination_num': coordination_num
        }

    def get_ifp_info(self, pdb_path: str, ligand_path: str) -> Dict:
        """Get IFP info using PDBProtein (no metal) and ligand file."""
        _, complex_fn = self.join_complex(ligand_path, pdb_path)
        ifp_info = self.get_complex_interaction_info(complex_fn)
        os.remove(complex_fn)
        return ifp_info

    def GET_IFP(
        self,
        ligand_sdf_path: Optional[str],
        protein_no_metal_pdb_path: str,
        protein_pdb_path: Optional[str] = None
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Main workflow: Analyze protein-ligand interactions with optional metal coordination.
        """
        if ligand_sdf_path is None:
            if protein_pdb_path is None:
                protein_no_metal = PDBProtein(protein_no_metal_pdb_path, if_H=self.if_H).to_dict_atom()
                pocket_n = len(protein_no_metal['element'])
            else:
                protein_metal = PDBProtein(protein_pdb_path, allow_het=self.allow_het_list, if_H=self.if_H).to_dict_atom()
                pocket_n = len(protein_metal['element'])
            ifp_no_metal_matrix = torch.zeros(pocket_n, 8, dtype=torch.float32)
            return ifp_no_metal_matrix, []

        protein_no_metal = PDBProtein(protein_no_metal_pdb_path, if_H=self.if_H).to_dict_atom()
        ligand = parse_sdf_file(ligand_sdf_path)
        ligand_info = {'pos': ligand['pos'], 'element': ligand['element']}
        protein_no_metal_info = {'pos': protein_no_metal['pos'], 'element': protein_no_metal['element']}
        ifp_no_metal = self.get_ifp_info(protein_no_metal_pdb_path, ligand_sdf_path)

        if protein_pdb_path is None:
            ligand_n = len(ligand['element'])
            pocket_n = len(protein_no_metal['element'])
            ifp_no_metal_matrix = torch.tensor(
                self._get_pocket_interaction_matrix(ligand_n, pocket_n, ifp_no_metal),
                dtype=torch.float32
            )
            return ifp_no_metal_matrix, []

        protein_metal = PDBProtein(protein_pdb_path, allow_het=self.allow_het_list, if_H=self.if_H).to_dict_atom()

        metal_info = [
            {'idx': i, 'element': protein_metal['element'][i], 'pos': protein_metal['pos'][i]}
            for i in range(len(protein_metal['element']))
            if protein_metal['element'][i] in self.metal_element
        ]
        metal_idx = [metal['idx'] for metal in metal_info]
        mapping = {}
        j = 0
        for i in range(len(protein_metal['element'])):
            if i not in metal_idx:
                mapping[j] = i
                j += 1

        for metal in metal_info:
            result = self.analyze_ligand_metal_coordination(metal, ligand_info, protein_no_metal_info)
            metal['if_interact'] = result['is_coordinated']

        MET = [metal['idx'] for metal in metal_info if metal['if_interact']]

        ligand_n = len(ligand['element'])
        pocket_n_metal = len(protein_metal['element'])
        ifp_no_metal_convert = {
            key: [mapping[x - ligand_n] + ligand_n for x in value if x - ligand_n in mapping]
            for key, value in ifp_no_metal.items()
        }
        ifp_no_metal_convert['MET'] = [idx + ligand_n for idx in MET]

        ifp_metal = torch.tensor(
            self._get_pocket_interaction_matrix(ligand_n, pocket_n_metal, ifp_no_metal_convert),
            dtype=torch.float32
        )
        return ifp_metal, metal_info

    def save_ifp_to_csv(
        self,
        ligand_sdf_path: Optional[str] = None,
        protein_no_metal_pdb_path: Optional[str] = None,
        protein_pdb_path: Optional[str] = None,
        filename: str = "ifp_info.csv"
    ) -> Tuple[pd.DataFrame, Dict, torch.Tensor]:
        """
        Save IFP information to CSV with atom metadata.
        """
        if protein_pdb_path is None:
            atom = PDBProtein(protein_no_metal_pdb_path, if_H=self.if_H).to_dict_atom_2()
        else:
            atom = PDBProtein(protein_pdb_path, allow_het=self.allow_het_list, if_H=self.if_H).to_dict_atom_2()

        ifp_info, _ = self.GET_IFP(ligand_sdf_path, protein_no_metal_pdb_path, protein_pdb_path)
        atom_name, atom2chain, res_idx = atom['atom_name'], atom['atom2chain'], atom['atom2residue_idx']
        res_type = [AA_NUMBER_NAME[idx] for idx in atom['atom_to_aa_type']]

        df = pd.DataFrame(ifp_info.numpy(), columns=INTERACTION_COLUMNS)
        df.insert(0, "Atom Name", atom_name)
        df.insert(0, "Residue Idxx", res_idx)
        df.insert(0, "Residue Type", res_type)
        df.insert(0, "Chain Name", atom2chain)

        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        df.to_csv(filename, index=False)
        print(f"IFP information saved to {filename}")
        return df, atom, ifp_info

    def save_ifp_filtered_comparison_to_csv(
        self,
        ref_ligand_sdf_path: str,
        gen_ligand_sdf_path: str,
        protein_no_metal_pdb_path: str,
        protein_pdb_path: Optional[str] = None,
        ref_ifp_info: Optional[torch.Tensor] = None,
        filename: str = "ifp_comparison_filtered.csv"
    ) -> Tuple[pd.DataFrame, float]:
        """
        Save filtered comparison of reference and generated ligand IFP.
        Only rows with at least one non-zero interaction are kept.
        """
        if protein_pdb_path is None:
            atom = PDBProtein(protein_no_metal_pdb_path, if_H=self.if_H).to_dict_atom_2()
        else:
            atom = PDBProtein(protein_pdb_path, allow_het=self.allow_het_list, if_H=self.if_H).to_dict_atom_2()

        if ref_ifp_info is None:
            ref_ifp_info, _ = self.GET_IFP(ref_ligand_sdf_path, protein_no_metal_pdb_path, protein_pdb_path)
        gen_ifp_info, _ = self.GET_IFP(gen_ligand_sdf_path, protein_no_metal_pdb_path, protein_pdb_path)
        ifp_score = get_ifp_score(ref_ifp_info, gen_ifp_info)

        df_ref = pd.DataFrame(ref_ifp_info.numpy(), columns=INTERACTION_COLUMNS)
        df_gen = pd.DataFrame(gen_ifp_info.numpy(), columns=INTERACTION_COLUMNS)

        atom_name = atom['atom_name']
        atom2chain = atom['atom2chain']
        res_idx = atom['atom2residue_idx']
        res_type = [AA_NUMBER_NAME[idx] for idx in atom['atom_to_aa_type']]

        df_combined = pd.DataFrame({
            "Chain Name": atom2chain,
            "Residue Type": res_type,
            "Residue Idxx": res_idx,
            "Atom Name": atom_name
        })

        for col in INTERACTION_COLUMNS:
            df_combined[f"{col} (ref)"] = df_ref[col]
            df_combined[f"{col} (gen)"] = df_gen[col]

        interaction_cols = [col for col in df_combined.columns if col.endswith("(ref)") or col.endswith("(gen)")]
        df_filtered = df_combined.loc[df_combined[interaction_cols].ne(0).any(axis=1)]

        dirname = os.path.dirname(filename)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)

        df_filtered.to_csv(filename, index=False)
        print(f"Filtered IFP comparison information saved to {filename}")
        return df_filtered, ifp_score

    @staticmethod
    def load_ifp_from_csv(filename: str = "ifp_info.csv") -> torch.Tensor:
        """
        Load IFP information from CSV file.
        """
        df = pd.read_csv(filename)
        ifp_info = torch.tensor(df[INTERACTION_COLUMNS].values, dtype=torch.float32)
        return ifp_info


# Example usage
if __name__ == '__main__':
    protein_pdb_path = '/home/worker/users/WYG/IFP-diffusion_v2/dev_src/examples/7ltg_ligand_pocket10.pdb'
    ligand_sdf_path = '/home/worker/users/WYG/IFP-diffusion_v2/dev_src/examples/7ltg_ligand.sdf'
    protein_no_metal_pdb_path = '/home/worker/users/WYG/IFP-diffusion_v2/dev_src/examples/7ltg_pocket_nometal.pdb'

    analyzer = IFPAnalyzer()
    ifp_metal_tensor, metal_info = analyzer.GET_IFP(ligand_sdf_path, protein_no_metal_pdb_path, protein_pdb_path)
    print(ifp_metal_tensor[111])