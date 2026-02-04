import argparse
import json
import logging
import os
import pickle
import random
import shutil
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import SDWriter
from torch_geometric.transforms import Compose
from tqdm import tqdm

# Suppress unnecessary warnings to keep logs clean
RDLogger.DisableLog('rdApp.warning')
logging.getLogger('plip').setLevel(logging.WARNING)

# Add parent directory to sys.path for custom module imports
sys.path.append('..')

# Custom imports grouped by category
from datasets.pl_data import ProteinLigandData, torchify_dict
from datasets.plip_2 import IFPAnalyzer, ALLOW_HET_LIST
from models.molopt_score_model import IFPDiff
from posebusters import PoseBusters
from models.sample_diffusion import sample_diffusion_ligand
from utils import reconstruct
from utils.data import parse_sdf_file
from utils.evaluation import scoring_func
from utils.evaluation.docking_vina import VinaDockingTask
from utils.misc import get_logger, seed_all
import utils.misc as misc
from utils.protein import PDBProtein
from utils.transforms import (
    FeaturizeLigandAtom,
    FeaturizeProteinAtom,
    get_atomic_number_from_index,
    is_aromatic_from_index,
)


# Initialize global logger and analysis tools
logger = get_logger('evaluate')
buster = PoseBusters(config="mol")  # Tool for validating generated poses
ifptool = IFPAnalyzer()  # Tool for Interaction Fingerprint (IFP) analysis


def pdb_to_pocket_data(
    pocket_nometal_path: str,
    ligand_path: str = None,
    pocket_path: str = None,
    ref_IFP: torch.Tensor = None,
    atom_type: str = 'prior',
    if_H: bool = True,
) -> tuple:
    """
    Convert PDB and SDF files to pocket data dictionary for model input.

    This function processes protein pocket and ligand files into a structured
    data format suitable for the diffusion model, including optional IFP.

    Args:
        pocket_nometal_path: Path to pocket PDB without metals.
        ligand_path: Path to ligand SDF file (optional).
        pocket_path: Path to full pocket PDB (optional).
        ref_IFP: Reference Interaction Fingerprint tensor (optional).
        atom_type: Atom type sampling mode (default: 'prior').
        if_H: Include hydrogens in processing (default: True).

    Returns:
        Tuple: ProteinLigandData object and pocket dictionary.
    """
    # Load pocket dictionary from PDB file
    pocket_dict = PDBProtein(
        pocket_path if pocket_path and pocket_path != "None" else pocket_nometal_path,
        allow_het=ALLOW_HET_LIST,
        if_H=if_H,
    ).to_dict_atom()

    # Parse ligand from SDF file
    ligand_dict_refer = parse_sdf_file(ligand_path)
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict=torchify_dict(ligand_dict_refer),
    )
    data.unimol_repr = None  # Clear any pre-existing UniMol representation

    # Set IFP info: use reference if provided, else compute
    if ref_IFP is None:
        ifp_tensor, _ = ifptool.GET_IFP(ligand_path, pocket_nometal_path, pocket_path)
        data.ifp_info = ifp_tensor.long()
    else:
        logger.info("Using reference IFP")
        data.ifp_info = ref_IFP.long()

    # Set ligand filename for reference
    base_path = ligand_path if ligand_path else pocket_nometal_path
    data.ligand_filename = os.path.join(*base_path.strip('/').split('/')[-2:])
    return data, pocket_dict


def load_pocket_data(crossdock_dir: str) -> list:
    """
    Load pocket data from CrossDocked test set pickle file.

    Args:
        crossdock_dir: Path to CrossDocked dataset pickle.

    Returns:
        List of pocket file paths.

    Raises:
        FileNotFoundError: If the dataset file is missing.
    """
    try:
        with open(crossdock_dir, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        logger.error(f"CrossDocked dataset not found at {crossdock_dir}")
        raise


def load_model(ckpt_path: str, device: torch.device) -> tuple:
    """
    Load the IFPDiff model and data transformation pipeline.

    Args:
        ckpt_path: Path to model checkpoint file.
        device: Torch device for model loading.

    Returns:
        Tuple: Loaded model and Compose transform.

    Raises:
        FileNotFoundError: If checkpoint file is missing.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except FileNotFoundError:
        logger.error(f"Model checkpoint not found at {ckpt_path}")
        raise

    # Initialize featurizers for protein and ligand atoms
    protein_featurizer = FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([protein_featurizer, ligand_featurizer])

    # Initialize and load model state
    model = IFPDiff(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    return model, transform


def get_initial_ref_interaction(ref_ifp_path: str, pocket_idx: int) -> torch.Tensor:
    """
    Retrieve initial reference interaction for a specific pocket.

    Args:
        ref_ifp_path: Path to reference IFP pickle file.
        pocket_idx: Index of the pocket in the file.

    Returns:
        Reference IFP tensor.

    Raises:
        FileNotFoundError: If reference file is missing.
        IndexError: If pocket index is invalid.
    """
    try:
        with open(ref_ifp_path, 'rb') as f:
            ref_interactions = pickle.load(f)
        return ref_interactions[pocket_idx]
    except FileNotFoundError:
        logger.error(f"Reference interactions file not found at {ref_ifp_path}")
        raise
    except IndexError:
        logger.error(f"Invalid pocket index: {pocket_idx}")
        raise


def generate_molecules(data, num_samples: int, model, device, config, args) -> tuple:
    """
    Generate ligand molecules using the diffusion sampling process.

    This function runs the diffusion model to sample ligand positions and types,
    then reconstructs RDKit molecules from the predictions.

    Args:
        data: Input pocket data.
        num_samples: Number of ligands to generate.
        model: Diffusion model.
        device: Torch device.
        config: Sampling configuration.
        args: Command-line arguments.

    Returns:
        Tuple: Generated molecules, SMILES list, reconstruction counts, pocket result dict.
    """
    # Prepare full protein positions for sampling
    full_protein_pos = torch.tensor(data['protein_pos'], dtype=torch.float32)
    print("config: ",config)
    all_pred_pos, all_pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
        model, data, num_samples,
        batch_size=args.batch_size, device=device,
        num_steps=config.num_steps,
        pos_only=False,
        center_pos_mode=config.center_pos_mode,
        sample_num_atoms=config.sample_num_atoms,
        energy_drift_opt=config.energy_drift,
        full_protein_pos=full_protein_pos,
    )

    # Store prediction trajectories in result dict
    pocket_result = {
        'data': data.to_dict(),
        'pred_ligand_pos': all_pred_pos,
        'pred_ligand_v': all_pred_v,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': pred_v_traj,
    }

    gen_mols = []
    smiles_list = []
    n_recon_success, n_complete = 0, 0
    for pred_pos, pred_v in zip(all_pred_pos, all_pred_v):
        pred_atom_type = get_atomic_number_from_index(pred_v, mode='add_aromatic')
        try:
            pred_aromatic = is_aromatic_from_index(pred_v, mode='add_aromatic')
            mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)  # Remove explicit hydrogens
            smiles = Chem.MolToSmiles(mol)
        except Exception as e:
            logger.warning(f"Molecule reconstruction failed: {e}")
            gen_mols.append(None)
            continue
        n_recon_success += 1
        if '.' in smiles:  # Check for disconnected components
            gen_mols.append(None)
            continue
        n_complete += 1
        gen_mols.append(mol)
        smiles_list.append(smiles)

    pocket_result['mols'] = gen_mols
    logger.info(f"Generated {n_complete} molecules, {n_recon_success} successfully reconstructed")
    return gen_mols, smiles_list, n_recon_success, n_complete, pocket_result


def calc_molecule_metrics(
    mol,
    pocket_path: str,
    ref_interaction,
    pocket_metal_path: str = None,
    score_components: dict = None,
) -> dict:
    """
    Compute evaluation metrics for a generated molecule.

    Metrics include chemical properties (QED, SA), docking scores (Vina),
    interaction similarity (IFP), and pose validation (PoseBusters).
    The combined_score is calculated based on user-specified components and weights.

    Args:
        mol: RDKit molecule.
        pocket_path: Pocket PDB path.
        ref_interaction: Reference IFP.
        pocket_metal_path: Pocket with metals path (optional).
        score_components: Dict of components and their weights, e.g., {'vina':1.5, 'qed':1.0, 'sa':1.0, 'posebuster':1.0}.

    Returns:
        Dict of computed metrics.
    """
    # Default to original components if not specified
    if score_components is None:
        score_components = {'vina': 1.5, 'qed': 1.0, 'posebuster': 1.0}

    # Compute basic chemical properties
    chem_results = scoring_func.get_chem(mol)
    qed = chem_results['qed']
    sa = chem_results['sa']

    # Perform Vina docking
    vina_task = VinaDockingTask.from_generated_mol_2(mol, pocket_path)
    score_only_results = vina_task.run(mode='score_only', exhaustiveness=30)
    minimize_results = vina_task.run(mode='minimize', exhaustiveness=30)
    vina_results = {
        'score_only': score_only_results,
        'minimize': minimize_results,
    }
    vina = vina_results['score_only'][0]['affinity']

    # Create temporary SDF for IFP calculations
    with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp_file:
        ligand_tmp_path = tmp_file.name
        with SDWriter(ligand_tmp_path) as sdf_writer:
            sdf_writer.write(mol)

    # Compute IFP score and new IFP
    ifp_score = scoring_func.get_ifp_score(
        ligand_tmp_path, ref_interaction,
        ref_protein_nometal_path=pocket_path,
        ref_protein_metal_path=pocket_metal_path,
    )
    new_ifp = scoring_func.get_ifp(
        ligand_tmp_path, pocket_path,
        ref_protein_metal_path=pocket_metal_path,
    )
    os.remove(ligand_tmp_path)  # Clean up temporary file

    
    true_ratio = 1.0
    # Calculate combined score based on specified components
    combined_score = 0.0
    if 'vina' in score_components:
        combined_score += - score_components['vina'] * vina / 12
    if 'qed' in score_components:
        combined_score += score_components['qed'] * qed
    if 'sa' in score_components:
        combined_score += score_components['sa'] * sa
    if 'posebuster' in score_components:
        # Validate pose with PoseBusters
        df_buster = buster.bust([mol], None, None)
        experiments_columns = df_buster.columns[2:]
        true_count = (df_buster[experiments_columns] == True).sum(axis=1).iloc[0]
        true_ratio = true_count / len(experiments_columns)
        combined_score += score_components['posebuster'] * true_ratio
    # Note: IFP score not included in combined_score by default; can add if needed

    return {
        'mol': mol,
        'new_ifp': new_ifp,
        'vina_results': vina_results,
        'qed': qed,
        'sa': sa,
        'ifp_score': ifp_score,
        'PoseBuster': true_ratio,
        'combined_score': combined_score,
    }


def evaluate_molecules(molecules, pocket_path: str, ref_interaction, pocket_metal_path: str, score_components: dict) -> list:
    """
    Evaluate a list of molecules in parallel using multiprocessing.

    Args:
        molecules: List of RDKit molecules.
        pocket_path: Pocket PDB path.
        ref_interaction: Reference IFP.
        pocket_metal_path: Pocket with metals path.
        score_components: Dict of score components and weights.

    Returns:
        List of metric dictionaries for valid molecules.
    """
    valid_mols = [mol for mol in molecules if mol is not None]
    with Pool() as pool:
        results = pool.starmap(
            calc_molecule_metrics,
            [(mol, pocket_path, ref_interaction, pocket_metal_path, score_components) for mol in valid_mols],
        )
    return [r for r in results if r is not None]


def filter_candidates_by_ifp(candidates: list, fix_ifp: list) -> list:
    """
    Filter candidates based on specific IFP positions being 1.

    Args:
        candidates: List of metric dicts.
        fix_ifp: List of (row, col) indices to check in IFP.

    Returns:
        Filtered list of candidates.
    """
    filtered = []
    for r in candidates:
        if all(r['new_ifp'][idx[0], idx[1]] == 1 for idx in fix_ifp):
            filtered.append(r)
    return filtered


def filter_candidates(results: list, fix_ifp: list = None) -> list:
    """
    Filter molecules that meet predefined minimum metric thresholds.

    If no candidates meet criteria, fall back to best by combined score.

    Args:
        results: List of metric dicts.
        fix_ifp: Optional list of fixed IFP indices.

    Returns:
        Filtered candidates.
    """
    candidates = [
        r for r in results
        if r['qed'] >= 0.25
        and r['sa'] >= 0.45
        and r['ifp_score'] >= 0.5
        and r['vina_results']['score_only'][0]['affinity'] <= -8.18
        and r['PoseBuster'] >= 0.5
    ]

    if fix_ifp:
        candidates = filter_candidates_by_ifp(candidates, fix_ifp)
        logger.warning(f"Filtered candidates by fix_ifp: {len(candidates)}")

    if not candidates:
        logger.warning("No candidates meet the minimum criteria")
        candidates = results
        if fix_ifp:
            candidates = filter_candidates_by_ifp(candidates, fix_ifp) or results
            logger.warning(f"Filtered candidates by fix_ifp: {len(candidates)}")
        candidates = sorted(candidates, key=lambda x: x['combined_score'], reverse=True)[:1]

    return candidates


def select_top_molecules(results: list, k: int, fix_ifp: list = None) -> tuple:
    """
    Select top k molecules by combined score after filtering.

    Args:
        results: List of metric dicts.
        k: Number of top molecules to select.
        fix_ifp: Optional fixed IFP indices.

    Returns:
        Tuple: Top results list, average combined score.
    """
    candidates = filter_candidates(results, fix_ifp)
    top_results = sorted(candidates, key=lambda x: x['combined_score'], reverse=True)[:k]
    avg_score = np.mean([r['combined_score'] for r in results]) if results else 0
    return top_results, avg_score


def update_reference_interaction(top_results: list) -> torch.Tensor:
    """
    Update reference IFP based on the best molecule's new IFP.

    Args:
        top_results: List of top metric dicts.

    Returns:
        New IFP tensor or None if no results.
    """
    if not top_results:
        logger.warning("No top results to update reference interaction")
        return None
    best_result = max(top_results, key=lambda x: x['combined_score'])
    return best_result['new_ifp']


def update_refer_ligand_path(top_results: list) -> str:
    """
    Create temporary SDF for the best molecule as new reference ligand.

    Args:
        top_results: List of top metric dicts.

    Returns:
        Path to temporary SDF or None if no results.
    """
    if not top_results:
        logger.warning("No top results to update reference ligand")
        return None
    best_result = max(top_results, key=lambda x: x['combined_score'])
    temp_sdf_path = "tmp/temp_ligand.sdf"
    os.makedirs(os.path.dirname(temp_sdf_path), exist_ok=True)
    with SDWriter(temp_sdf_path) as writer:
        writer.write(best_result['mol'])
    return temp_sdf_path


def print_evaluation_info(results: list, iteration: int, posebuster_pass: float = None) -> dict:
    """
    Log average metrics for a set of molecules.

    Args:
        results: List of metric dicts.
        iteration: Current optimization iteration.
        posebuster_pass: Optional full PoseBusters pass rate.

    Returns:
        Dict of average metrics.
    """
    if not results:
        logger.info("No molecules to evaluate")
        return {}

    avg_vina_score = np.mean([s['vina_results']['score_only'][0]['affinity'] for s in results])
    avg_qed = np.mean([s['qed'] for s in results])
    avg_sa = np.mean([s['sa'] for s in results])
    avg_sim = np.mean([s['ifp_score'] for s in results])
    avg_combined = np.mean([s['combined_score'] for s in results])
    avg_posebuster = np.mean([s['PoseBuster'] for s in results])

    logger.info(f"Evaluation of {len(results)} molecules:")
    logger.info(f"  Avg QED: {avg_qed:.3f}, Avg SA: {avg_sa:.3f}, Avg Similarity: {avg_sim:.3f}")
    logger.info(f"  Avg Vina_score: {avg_vina_score:.3f}")
    logger.info(f"  Avg PoseBuster: {avg_posebuster:.3f}")
    logger.info(f"  Avg Combined Score: {avg_combined:.3f}")
    if posebuster_pass is not None:
        logger.info(f"  Avg PoseBuster pass rate: {posebuster_pass:.3f}")

    return {
        'iteration': iteration,
        'avg_vina_score': avg_vina_score,
        'avg_qed': avg_qed,
        'avg_sa': avg_sa,
        'avg_sim': avg_sim,
        'avg_PoseBuster': avg_posebuster,
        'PoseBuster_pass': posebuster_pass,
        'avg_combined': avg_combined,
    }

def save_all_pocket_result(pocket_idx: int, result: list, output_dir: str):
    """
    Save per-pocket results to a pickle file.

    Args:
        pocket_idx: Pocket index.
        result: List of iteration results.
        output_dir: Base output directory.
    """
    final_path = os.path.join(output_dir, "all_pocket_result", f"{pocket_idx}.pkl")
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    try:
        with open(final_path, 'wb') as f:
            pickle.dump(result, f)
        logger.info(f"Saved single pocket result to: {final_path}")
    except Exception as e:
        logger.error(f"Failed to save single pocket result: {e}")

def save_single_pocket_result(pocket_idx: int, result: dict, output_dir: str, iteration: int):
    """
    Save per-pocket results to a pickle file.

    Args:
        pocket_idx: Pocket index.
        result: Dict of iteration result.
        output_dir: Base output directory.
    """
    final_path = os.path.join(output_dir, "single_pocket_result", f"{pocket_idx}_{iteration}","sample.pt")
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    try:
        torch.save(result,final_path)
        logger.info(f"Saved single pocket result to: {final_path}")
    except Exception as e:
        logger.error(f"Failed to save single pocket result: {e}")

def save_iteration_result(pocket_idx: int, result: dict, output_dir: str):
    """
    Append iteration result to per-pocket JSON file.

    Args:
        pocket_idx: Pocket index.
        result: Iteration metric dict.
        output_dir: Base output directory.
    """
    final_path = os.path.join(output_dir, 'single_pocket_result', f"{pocket_idx}.json")
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    try:
        existing_results = []
        if os.path.exists(final_path):
            with open(final_path, 'r') as f:
                existing_results = json.load(f)
        existing_results.append(result)
        with open(final_path, 'w') as f:
            json.dump(existing_results, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save iteration result: {e}")


def compute_final_statistics(all_results: list):
    """
    Compute and log aggregate statistics across all processed pockets.

    Args:
        all_results: List of final pocket results.
    """
    if not all_results:
        logger.info("No results to compute statistics")
        return

    statistics = {
        'vina_score': [],
        'qed': [],
        'sa': [],
        'sim': [],
        'PoseBuster': [],
        'PoseBusters_pass': [],
        'combined_score': [],
    }

    for result in all_results:
        try:
            results = result['results']
            statistics['vina_score'].append(np.mean([s['vina_results']['score_only'][0]['affinity'] for s in results]))
            statistics['qed'].append(np.mean([s['qed'] for s in results]))
            statistics['sa'].append(np.mean([s['sa'] for s in results]))
            statistics['sim'].append(np.mean([s['ifp_score'] for s in results]))
            statistics['combined_score'].append(np.mean([s['combined_score'] for s in results]))
            statistics['PoseBuster'].append(np.mean([s['PoseBuster'] for s in results]))
            statistics['PoseBusters_pass'].append(result['PoseBusters_pass'])
        except KeyError:
            logger.warning("Missing key in results, skipping this result")
            continue

    logger.info("Final statistics for all pockets:")
    for key, values in statistics.items():
        avg = np.mean(values)
        median = np.median(values)
        logger.info(f"  Avg {key.replace('_', ' ').title()}: {avg:.3f}, Median {key.replace('_', ' ').title()}: {median:.3f}")


def calculate_posebusters_pass(mols: list) -> float:
    """
    Calculate the ratio of molecules that fully pass all PoseBusters checks.

    Args:
        mols: List of RDKit molecules.

    Returns:
        Pass ratio (0.0 to 1.0).
    """
    valid_mols = [mol for mol in mols if mol is not None]
    df_buster = buster.bust(valid_mols, None, None)
    experiments_columns = df_buster.columns[2:]
    count_all_true = (df_buster[experiments_columns].all(axis=1)).sum()
    return count_all_true / len(df_buster) if len(df_buster) > 0 else 0.0


def load_input_data(config_path: str) -> list:
    """
    Load pocket input data from JSON file.

    Args:
        config_path: Path to input JSON.

    Returns:
        List of pocket dicts.
    """
    with open(config_path, 'r') as f:
        pocket_files = json.load(f)

    input_json_dir = os.path.dirname(os.path.abspath(args.input_json))
    for p in pocket_files:
        for k,v in p.items():
            # 仅当v为str类型时才进行os.path.isabs与os.path.join等操作
            if not isinstance(v, str):
                continue
            if not os.path.isabs(v):
                p[k] = os.path.normpath(os.path.join(input_json_dir,v))
    return pocket_files


def parse_score_components(score_str: str) -> dict:
    """
    Parse the score_components argument string into a dict of components and weights.

    Example: "vina:1.5,qed:1.0,sa:1.0,posebuster:1.0" -> {'vina':1.5, 'qed':1.0, 'sa':1.0, 'posebuster':1.0}

    Args:
        score_str: Comma-separated string of component:weight pairs.

    Returns:
        Dict of components to weights.
    """
    components = {}
    if score_str:
        pairs = score_str.split(',')
        for pair in pairs:
            if ':' in pair:
                comp, weight = pair.split(':')
                components[comp.strip()] = float(weight.strip())
    return components


def optimize_pocket(
    pocket_idx: int,
    pocket_file: dict,
    model,
    transform,
    device,
    config,
    args,
) -> dict:
    """
    Run iterative optimization for a single pocket.

    Generates molecules, evaluates them, and updates references until no improvement
    or max iterations reached. If max_iterations=1, runs only once without iteration.

    Args:
        pocket_idx: Pocket index.
        pocket_file: Dict with pocket paths and references.
        model: Diffusion model.
        transform: Data transformer.
        device: Torch device.
        config: Sampling config.
        args: Arguments.

    Returns:
        Final pocket result dict.
    """
    start_time = time.time()
    pocket_path = pocket_file['pdb_path']
    pocket_path_metal = pocket_file['pdb_metal_path']
    ligand_path = pocket_file['ref_ligand_sdf']
    ref_interaction = ifptool.load_ifp_from_csv(pocket_file['ref_ifp'])
    fix_ifp = pocket_file['fix_ifp']

    best_ref_interaction = ref_interaction
    save_ref_interaction = ref_interaction
    best_refer_ligand_path = ligand_path
    best_score = -float('inf')
    save_sample = None
    iteration = 0
    single_pocket_result = []

    # If max_iterations == 1, we run the loop once without updating references
    max_iters = args.max_iterations if args.max_iterations > 1 else 1
    while iteration < max_iters:
        logger.info(f"\nIteration {iteration + 1}/{max_iters}")
        # Prepare data for current iteration
        data, _ = pdb_to_pocket_data(
            pocket_path,
            ligand_path=best_refer_ligand_path,
            ref_IFP=best_ref_interaction,
            pocket_path=pocket_path_metal,
        )
        data = transform(data)
        molecules, _, _, _, save_result = generate_molecules(data, args.sample_size, model, device, config, args)

        # Evaluate generated molecules with custom score components
        results = evaluate_molecules(molecules, pocket_path, best_ref_interaction, pocket_metal_path=pocket_path_metal, score_components=args.score_components)
        posebusters_pass = calculate_posebusters_pass(molecules)
        pocket_result = {
            'iteration': iteration,
            'PoseBusters_pass': posebusters_pass,
            'results': results,
            'pdb_path': pocket_file['pdb_path'],
            'Refer_IFP': best_ref_interaction,
            'fix_ifp': fix_ifp,
        }

        # Select top molecules
        top_results, avg_combined_score = select_top_molecules(results, args.top_k, fix_ifp=fix_ifp)
        pocket_result['top_result'] = top_results

        logger.info("Evaluation results:")
        print_evaluation_info(results, iteration, posebusters_pass)
        logger.info("Top results:")
        print_evaluation_info(top_results, iteration)

        # If max_iterations == 1, skip improvement check and updates
        if args.max_iterations == 1:
            single_pocket_result.append(pocket_result)
            break

        # Check for improvement
        if avg_combined_score <= best_score:
            logger.info(f"No improvement detected. Stopping at iteration {iteration + 1}")
            break

        single_pocket_result.append(pocket_result)
        best_score = avg_combined_score
        best_result = [random.choice(top_results) if top_results else None]
        new_ref_interaction = update_reference_interaction(best_result)
        best_refer_ligand_path = update_refer_ligand_path(best_result)

        if new_ref_interaction is not None:
            best_ref_interaction = new_ref_interaction
        else:
            logger.info(f"No new reference interaction found. Stopping at iteration {iteration + 1}")
            break
        # Save the results of different iterations of a pocket
        save_single_pocket_result(pocket_idx, pocket_result, args.output_dir, iteration)
        iteration += 1

    # Save final results
    save_all_pocket_result(pocket_idx, single_pocket_result, args.output_dir)

    end_time = time.time()
    logger.info(f"Processing time: {end_time - start_time:.2f} seconds")
    return pocket_result


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Molecular optimization script")
    parser.add_argument("--output_dir", default="../../InterOpt_output_v2", help="Output directory")
    parser.add_argument("--max_iterations", type=int, default=1, help="Maximum number of iterations (set to 1 for no iteration optimization)")
    parser.add_argument("--sample_size", type=int, default=20, help="Number of molecules to sample")
    parser.add_argument("--top_k", type=int, default=1, help="Number of top molecules to select")
    parser.add_argument("--device", default="cuda", help="Computing device")
    parser.add_argument("--ckpt_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", required=True, help="Config file for model sampling")
    parser.add_argument("--input_json", required=True, type=str, help="Path to input data JSON")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for optimization")
    parser.add_argument("--seed", type=int, default=2021, help="Random seed")
    parser.add_argument("--overwrite", type=int, default=False, help="Overwrite existing output files")
    parser.add_argument("--score_components", type=str, default="vina:1.5,qed:1.0,posebuster:1.0",
                        help="Comma-separated component:weight pairs for combined_score, e.g., 'vina:1.5,qed:1.0,sa:1.0,posebuster:1.0'")
    args = parser.parse_args()

    # Parse score components
    args.score_components = parse_score_components(args.score_components)

    # Setup output directory and seed
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)

    # Load input data and model
    pocket_files = load_input_data(args.input_json)
    
    model, transform = load_model(args.ckpt_path, torch.device(args.device))
    config = misc.load_config(args.config)  # Load sampling configuration

    all_results = []
    for pocket_idx, pocket_file in enumerate(tqdm(pocket_files)):
        final_path = os.path.join(args.output_dir, 'all_pocket_result', f"{pocket_idx}.pkl")
        print(final_path)
        if os.path.exists(final_path) and not args.overwrite:
            logger.info(f"File {final_path} already exists, skipping...")
            continue

        logger.info(f"\n=== Processing pocket {pocket_idx + 1}/{len(pocket_files)} ===")
        pocket_result = optimize_pocket(pocket_idx, pocket_file, model, transform, torch.device(args.device), config, args)
        all_results.append(pocket_result)

    # Compute and log final statistics
    compute_final_statistics(all_results)