from collections import Counter
from copy import deepcopy

import numpy as np
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
from rdkit.Chem.FilterCatalog import *
from rdkit.Chem.QED import qed

from utils.evaluation.sascorer import compute_sa_score
from utils.protein import PDBProtein
from datasets.plip_2 import IFPAnalyzer,ALLOW_HET_LIST

# from datasets.crossdock_plip_dataset import join_complex,get_complex_interaction_info, _get_pocket_interaction_matrix

import torch
import itertools
from utils.data import PDBProtein, parse_sdf_file
from rdkit.DataStructs import TanimotoSimilarity

def is_pains(mol):
    params_pain = FilterCatalogParams()
    params_pain.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    catalog_pain = FilterCatalog(params_pain)
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    entry = catalog_pain.GetFirstMatch(mol)
    if entry is None:
        return False
    else:
        return True


def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    logp = get_logp(mol)
    rule_4 = (logp >= -2) & (logp <= 5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])


def get_basic(mol):
    n_atoms = len(mol.GetAtoms())
    n_bonds = len(mol.GetBonds())
    n_rings = len(Chem.GetSymmSSSR(mol))
    weight = Descriptors.ExactMolWt(mol)
    return n_atoms, n_bonds, n_rings, weight


def get_rdkit_rmsd(mol, n_conf=20, random_seed=42):
    """
    Calculate the alignment of generated mol and rdkit predicted mol
    Return the rmsd (max, min, median) of the `n_conf` rdkit conformers
    """
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    mol3d = Chem.AddHs(mol)
    rmsd_list = []
    # predict 3d
    try:
        confIds = AllChem.EmbedMultipleConfs(mol3d, n_conf, randomSeed=random_seed)
        for confId in confIds:
            AllChem.UFFOptimizeMolecule(mol3d, confId=confId)
            rmsd = Chem.rdMolAlign.GetBestRMS(mol, mol3d, refId=confId)
            rmsd_list.append(rmsd)
        # mol3d = Chem.RemoveHs(mol3d)
        rmsd_list = np.array(rmsd_list)
        return [np.max(rmsd_list), np.min(rmsd_list), np.median(rmsd_list)]
    except:
        return [np.nan, np.nan, np.nan]


def get_logp(mol):
    return Crippen.MolLogP(mol)


def get_chem(mol):
    qed_score = qed(mol)
    sa_score = compute_sa_score(mol)
    logp_score = get_logp(mol)
    lipinski_score = obey_lipinski(mol)
    ring_info = mol.GetRingInfo()
    ring_size = Counter([len(r) for r in ring_info.AtomRings()])
    # hacc_score = Lipinski.NumHAcceptors(mol)
    # hdon_score = Lipinski.NumHDonors(mol)

    return {
        'qed': qed_score,
        'sa': sa_score,
        'logp': logp_score,
        'lipinski': lipinski_score,
        'ring_size': ring_size
    }


def get_molecule_force_field(mol, conf_id=None, force_field='mmff', **kwargs):
    """
    Get a force field for a molecule.
    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    conf_id : int, optional
        ID of the conformer to associate with the force field.
    force_field : str, optional
        Force Field name.
    kwargs : dict, optional
        Keyword arguments for force field constructor.
    """
    if force_field == 'uff':
        ff = AllChem.UFFGetMoleculeForceField(
            mol, confId=conf_id, **kwargs)
    elif force_field.startswith('mmff'):
        AllChem.MMFFSanitizeMolecule(mol)
        mmff_props = AllChem.MMFFGetMoleculeProperties(
            mol, mmffVariant=force_field)
        ff = AllChem.MMFFGetMoleculeForceField(
            mol, mmff_props, confId=conf_id, **kwargs)
    else:
        raise ValueError("Invalid force_field {}".format(force_field))
    return ff


def get_conformer_energies(mol, force_field='mmff'):
    """
    Calculate conformer energies.
    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    force_field : str, optional
        Force Field name.
    Returns
    -------
    energies : array_like
        Minimized conformer energies.
    """
    energies = []
    for conf in mol.GetConformers():
        ff = get_molecule_force_field(mol, conf_id=conf.GetId(), force_field=force_field)
        energy = ff.CalcEnergy()
        energies.append(energy)
    energies = np.asarray(energies, dtype=float)
    return energies

# ------------------------------------------
def get_diversity(mols):
    fgs = [Chem.RDKFingerprint(mol) for mol in mols]
    all_fg_pairs = list(itertools.combinations(fgs, 2))
    similarity_list = []
    for fg1, fg2 in all_fg_pairs:
        similarity_list.append(TanimotoSimilarity(fg1, fg2))
    return 1 - np.mean(similarity_list),1-np.median(similarity_list)


# ------------------------------------------
def get_ifp_score(ligand_tmp_path, ref_ifp, ref_protein_nometal_path, ref_protein_metal_path = None, G=None, eps=1e-8):

    if G is None:
        analyzer = IFPAnalyzer()
        G, _ = analyzer.GET_IFP(ligand_tmp_path, ref_protein_nometal_path, ref_protein_metal_path)
    
    similarities = []
    num = 0
    for p_vec, g_vec in zip(ref_ifp.float(), G):
        mask = (p_vec > 0)
        if mask.sum() == 0:
            continue
        else:
            p_required = p_vec[mask]
            g_required = g_vec[mask]
            dot_prod = torch.dot(p_required, g_required)
            norm_p = torch.norm(p_required)
            norm_g = torch.norm(g_required)
            sim = dot_prod / (norm_p * norm_g + eps)
            similarities.append(sim)
            num += 1

    # Step 5: 返回最终的相似度
    if num == 0:
        return 1
    return sum(similarities) / num

def get_ifp(ligand_tmp_path, ref_protein_nometal_path, ref_protein_metal_path = None):

    analyzer = IFPAnalyzer()
    G, _ = analyzer.GET_IFP(ligand_tmp_path, ref_protein_nometal_path, ref_protein_metal_path)
    return G