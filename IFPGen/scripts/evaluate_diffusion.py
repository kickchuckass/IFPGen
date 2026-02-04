import argparse
import os
import sys
import tempfile
import pickle
from collections import Counter
from glob import glob

import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from tqdm.auto import tqdm

# Add parent directory to sys.path for custom imports
sys.path.append('..')

from utils import misc, reconstruct, transforms
from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask

def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')

def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = sum(1 for counter in all_ring_sizes if ring_size in counter)
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')

def main(args):
    misc.seed_all(args.seed)
    result_path = os.path.join(args.sample_path, 'eval_results')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    # Load generated data
    results_fn_list = glob(os.path.join(args.sample_path, '**', '*sample*.pt'), recursive=True)
    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[:args.eval_num_examples]
    num_examples = len(results_fn_list)
    logger.info(f'Loaded {num_examples} generated examples.')

    # Load test pocket list
    with open(args.test_pl_path, 'rb') as f:
        test_pl = pickle.load(f)

    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_eval_success, n_complete = 0, 0, 0
    results = []
    results_split_pocket = []
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()

    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Evaluating')):
        r = torch.load(r_name)
        all_pred_ligand_pos = r['pred_ligand_pos_traj']
        all_pred_ligand_v = r['pred_ligand_v_traj']
        num_samples += len(all_pred_ligand_pos)
        
        ref_ifp = r['data']['ifp_info']

        # Get reference protein paths
        if num_examples > 1:
            pocket_name = r_name.split('/')[-2][7:]
            ref_protein_entry = next((i for i in test_pl if i[0].split('/')[-1].split('.')[0] == pocket_name), None)
            if ref_protein_entry is None:
                logger.warning(f'No reference protein found for {pocket_name}')
                continue
            ref_protein_nometal = ref_protein_entry[0]
            ref_protein_metal = ref_protein_entry[2] if len(ref_protein_entry) == 3 else None
            ref_protein = ref_protein_metal or ref_protein_nometal
        else:
            ref_protein_nometal = args.single_protein_nometal
            ref_protein_metal = args.single_protein_metal
            ref_protein = ref_protein_metal or ref_protein_nometal

        pocket_results = []
        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
            pred_pos = pred_pos[args.eval_step]
            pred_v = pred_v[args.eval_step]

            # Stability check
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)
            all_atom_types.update(pred_atom_type)
            r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]

            pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
            all_pair_dist.extend(pair_dist)

            # Reconstruction
            try:
                pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                mol = Chem.RemoveHs(mol)
                smiles = Chem.MolToSmiles(mol)
            except reconstruct.MolReconsError:
                if args.verbose:
                    logger.warning(f'Reconstruction failed for {example_idx}_{sample_idx}')
                continue
            n_recon_success += 1

            if '.' in smiles:
                continue
            n_complete += 1

            # Chemical and docking evaluation
            try:
                chem_results = scoring_func.get_chem(mol)
                if args.docking_mode == 'qvina':
                    vina_task = QVinaDockingTask.from_generated_mol(
                        mol, r['data']['ligand_filename'], protein_root=args.protein_root
                    )
                    vina_results = vina_task.run_sync()
                elif args.docking_mode in ['vina_score', 'vina_dock']:
                    vina_task = VinaDockingTask.from_generated_mol_2(mol, ref_protein)
                    score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
                    minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
                    vina_results = {
                        'score_only': score_only_results,
                        'minimize': minimize_results
                    }
                    if args.docking_mode == 'vina_dock':
                        docking_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
                        vina_results['dock'] = docking_results
                else:
                    vina_results = None

                n_eval_success += 1
            except Exception as e:
                if args.verbose:
                    logger.warning(f'Evaluation failed for {example_idx}_{sample_idx}: {str(e)}')
                continue

            # IFP Score
            with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp:
                ligand_tmp_path = tmp.name
                Chem.MolToMolFile(mol, ligand_tmp_path)
            ifp_score = scoring_func.get_ifp_score(
                ligand_tmp_path, ref_ifp,
                ref_protein_nometal_path=ref_protein_nometal,
                ref_protein_metal_path=ref_protein_metal
            )
            os.remove(ligand_tmp_path)

            bond_dist = eval_bond_length.bond_distance_from_mol(mol)
            all_bond_dist.extend(bond_dist)

            success_pair_dist.extend(pair_dist)
            success_atom_types.update(pred_atom_type)

            result = {
                'mol': mol,
                'smiles': smiles,
                'ligand_filename': r['data']['ligand_filename'],
                'pred_pos': pred_pos,
                'pred_v': pred_v,
                'chem_results': chem_results,
                'vina': vina_results,
                'ifp_score': ifp_score,
                'ref_protein': ref_protein,
            }
            results.append(result)
            pocket_results.append(result)

        results_split_pocket.append(pocket_results)

    logger.info(f'Evaluation complete! Processed {num_samples} samples.')

    # Validity metrics
    validity_dict = {
        'mol_stable': all_mol_stable / num_samples if num_samples > 0 else 0,
        'atm_stable': all_atom_stable / all_n_atom if all_n_atom > 0 else 0,
        'recon_success': n_recon_success / num_samples if num_samples > 0 else 0,
        'eval_success': n_eval_success / num_samples if num_samples > 0 else 0,
        'complete': n_complete / num_samples if num_samples > 0 else 0
    }
    logger.info('Validity metrics:')
    print_dict(validity_dict, logger)

    # Bond length metrics
    c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
    logger.info('JS bond distances of complete mols:')
    print_dict(c_bond_length_dict, logger)

    success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
    success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)
    logger.info('Success JS metrics:')
    print_dict(success_js_metrics, logger)

    # Atom type distribution
    atom_type_js = eval_atom_type.eval_atom_type_distribution(success_atom_types)
    logger.info(f'Atom type JS: {atom_type_js:.4f}')

    if args.save:
        eval_bond_length.plot_distance_hist(
            success_pair_length_profile,
            metrics=success_js_metrics,
            save_path=os.path.join(result_path, f'pair_dist_hist_{args.eval_step}.png')
        )

    logger.info(f'Reconstructed: {n_recon_success}, Complete: {n_complete}, Evaluated: {len(results)}')

    # Diversity
    mols = [r['mol'] for r in results]
    diversity_mean, diversity_median = scoring_func.get_diversity(mols)
    logger.info(f'Diversity Mean: {diversity_mean:.3f}')
    logger.info(f'Diversity Median: {diversity_median:.3f}')

    # Chemical metrics
    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    ifp = [r['ifp_score'] for r in results]
    logger.info(f'QED: Mean {np.mean(qed):.3f}, Median {np.median(qed):.3f}')
    logger.info(f'SA: Mean {np.mean(sa):.3f}, Median {np.median(sa):.3f}')
    logger.info(f'IFP: Mean {np.mean(ifp):.3f}, Median {np.median(ifp):.3f}')

    # Docking metrics
    if args.docking_mode == 'qvina':
        vina = [r['vina'][0]['affinity'] for r in results]
        logger.info(f'Vina: Mean {np.mean(vina):.3f}, Median {np.median(vina):.3f}')
    elif args.docking_mode in ['vina_score', 'vina_dock']:
        vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
        vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]
        logger.info(f'Vina Score: Mean {np.mean(vina_score_only):.3f}, Median {np.median(vina_score_only):.3f}')
        logger.info(f'Vina Min: Mean {np.mean(vina_min):.3f}, Median {np.median(vina_min):.3f}')
        if args.docking_mode == 'vina_dock':
            vina_dock = [r['vina']['dock'][0]['affinity'] for r in results]
            logger.info(f'Vina Dock: Mean {np.mean(vina_dock):.3f}, Median {np.median(vina_dock):.3f}')

    # Ring distribution
    print_ring_ratio([r['chem_results']['ring_size'] for r in results], logger)

    # Save results
    torch.save({
        'stability': validity_dict,
        'bond_length': all_bond_dist,
        'all_results': results
    }, os.path.join(result_path, 'metrics_.pt'))

    torch.save(results_split_pocket, os.path.join(result_path, 'results_split_pocket.pt'))
    logger.info('Evaluation done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate generated molecular samples')
    parser.add_argument('sample_path', type=str, help='Path to sample directory')
    parser.add_argument('--test_pl_path', type=str, required=True, help='Path to test pocket list pickle')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--eval_step', type=int, default=-1, help='Evaluation step index')
    parser.add_argument('--eval_num_examples', type=int, default=None, help='Number of examples to evaluate')
    parser.add_argument('--save', action='store_true', default=True, help='Save plots and results')
    parser.add_argument('--protein_root', type=str, default='./data/crossdocked_v1.1_rmsd1.0', help='Protein root directory for QVina')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic', help='Atom encoding mode')
    parser.add_argument('--docking_mode', type=str, choices=['qvina', 'vina_score', 'vina_dock', 'none'], required=True, help='Docking mode')
    parser.add_argument('--exhaustiveness', type=int, default=30, help='Exhaustiveness for Vina docking')
    parser.add_argument('--single_protein_nometal', type=str, default=None, help='Single protein without metal')
    parser.add_argument('--single_protein_metal', type=str, default=None, help='Single protein with metal')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed')
    args = parser.parse_args()
    main(args)

    # test_pl_path = '/home/worker/users/WYG/IFP-diffusion_v2/datasets/2511_combine_pdbcrossdock/crossdock_test.pkl'
    # test_pl_path = '/home/worker/users/WYG/IFP-diffusion_v2/datasets/Medba_dataset/medba_test.pkl'
    # test_pl_path = '/home/worker/users/WYG/IFP-diffusion_v2/datasets/Medba_dataset/medba_test_10_list.pkl'
    # test_pl_path = '/home/worker/users/WYG/IFP-diffusion_v2/datasets/Medba_dataset/medba_test_10_list_v2_NI_MN.pkl'