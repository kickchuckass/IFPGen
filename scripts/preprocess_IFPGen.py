import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import List, Dict, Optional

from rdkit import Chem, RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.warning')

# Add parent directory to sys.path for custom imports
sys.path.append('..')

from datasets.plip_2 import IFPAnalyzer

# Initialize IFP analyzer
ifptool = IFPAnalyzer()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def preprocess_ifp(
    pdb_nometal_path: str,
    pdb_path: str,
    sdf_path: str,
    output_ifp_path: str,
) -> str:
    """
    Generate reference Interaction Fingerprint (IFP) from PDB and SDF files.

    Args:
        pdb_nometal_path: Path to PDB file without metals.
        pdb_path: Path to full PDB file.
        sdf_path: Path to SDF file.
        output_ifp_path: Path to output IFP CSV file.

    Returns:
        Path to generated IFP file.

    Raises:
        Exception: If IFP generation fails.
    """
    try:
        ifptool.save_ifp_to_csv(
            ligand_sdf_path=sdf_path,
            protein_no_metal_pdb_path=pdb_nometal_path,
            protein_pdb_path=pdb_path,
            filename=output_ifp_path,
        )
        logger.info(f"Generated IFP file at: {output_ifp_path}")
        return output_ifp_path
    except Exception as e:
        logger.error(f"Failed to generate IFP for {pdb_path} and {sdf_path}: {e}")
        raise


def validate_file(path: str, file_type: str) -> bool:
    """
    Validate if a file exists and has the correct extension.

    Args:
        path: File path.
        file_type: 'pdb' or 'sdf'.

    Returns:
        True if valid, False otherwise.
    """
    if not os.path.exists(path):
        logger.error(f"{file_type.upper()} file not found: {path}")
        return False
    if file_type == 'pdb' and not path.endswith('.pdb'):
        logger.error(f"Invalid PDB file: {path} (must have .pdb extension)")
        return False
    if file_type == 'sdf' and not path.endswith('.sdf'):
        logger.error(f"Invalid SDF file: {path} (must have .sdf extension)")
        return False
    return True


def process_single_input(
    pdb_path: str,
    ref_ligand_path: Optional[str] = None,
    pdb_metal_path: Optional[str] = None,
    output_dir: str = "output",
) -> Dict:
    """
    Process a single input to generate a JSON config item.

    Args:
        pdb_path: Path to PDB file.
        ref_ligand_path: Path to reference ligand SDF file (optional).
        pdb_metal_path: Path to PDB file with metals (optional).
        output_dir: Output directory for IFP files.

    Returns:
        JSON config dictionary.

    Raises:
        ValueError: If PDB file is invalid.
    """
    if not validate_file(pdb_path, 'pdb'):
        raise ValueError(f"Invalid PDB file: {pdb_path}")

    ref_ifp = None
    if ref_ligand_path:
        if not validate_file(ref_ligand_path, 'sdf'):
            raise ValueError(f"Invalid SDF file: {ref_ligand_path}")
        output_ifp_path = os.path.join(
            output_dir, f"ifp_{os.path.basename(pdb_path).replace('.pdb', '')}.csv"
        )
        os.makedirs(output_dir, exist_ok=True)
        ref_ifp = preprocess_ifp(pdb_path, pdb_metal_path or pdb_path, ref_ligand_path, output_ifp_path)

    return {
        "pocket_idx": 0,  # Default index for single input
        "pdb_path": pdb_path,
        "ref_ligand_sdf": ref_ligand_path,
        "pdb_metal_path": pdb_metal_path,
        "ref_ifp": ref_ifp,
        "fix_ifp": None,
    }


def load_list_input(list_file: str) -> List[List[Optional[str]]]:
    """
    Load multiple inputs from a list file (JSON or PKL).

    Args:
        list_file: Path to list file.

    Returns:
        List of [pdb_path, ref_ligand_path (optional), pdb_metal_path (optional)].

    Raises:
        ValueError: If file format is unsupported.
    """
    if list_file.endswith('.json'):
        with open(list_file, 'r') as f:
            data = json.load(f)
        return [
            [item['pdb_path'], item.get('ref_ligand_path'), item.get('pdb_metal_path')]
            for item in data
        ]
    elif list_file.endswith('.pkl'):
        with open(list_file, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported list file format: {list_file}. Must be .json or .pkl.")


def process_list_input(list_file: str, output_dir: str = "output") -> List[Dict]:
    """
    Process list input to generate list of JSON config items.

    Args:
        list_file: Path to list file.
        output_dir: Output directory for IFP files.

    Returns:
        List of JSON config dictionaries.
    """
    inputs = load_list_input(list_file)
    config_items = []
    for idx, items in enumerate(inputs):
        if len(items) == 3:
            pdb_path, ref_ligand_path, pdb_metal_path = items
        elif len(items) == 2:
            pdb_path, ref_ligand_path = items
            pdb_metal_path = None
        else:
            logger.warning(f"Skipping invalid input at index {idx}: {items}")
            continue

        if not validate_file(pdb_path, 'pdb'):
            logger.warning(f"Skipping invalid PDB file: {pdb_path}")
            continue

        ref_ifp = None
        if ref_ligand_path:
            if not validate_file(ref_ligand_path, 'sdf'):
                logger.warning(f"Skipping invalid SDF file: {ref_ligand_path}")
                continue
            output_ifp_path = os.path.join(
                output_dir, f"ifp_{idx}_{os.path.basename(pdb_path).replace('.pdb', '')}.csv"
            )
            os.makedirs(output_dir, exist_ok=True)
            ref_ifp = preprocess_ifp(pdb_path, pdb_metal_path or pdb_path, ref_ligand_path, output_ifp_path)

        config_items.append({
            "pocket_idx": idx,
            "pdb_path": pdb_path,
            "ref_ligand_sdf": ref_ligand_path,
            "pdb_metal_path": pdb_metal_path,
            "ref_ifp": ref_ifp,
            "fix_ifp": None,
        })
    return config_items


def save_config(config_data: List[Dict], output_path: str):
    """
    Save config data to JSON file.

    Args:
        config_data: List of config dictionaries.
        output_path: Path to output JSON file.
    """
    with open(output_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    logger.info(f"Saved configuration to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSON configuration for molecular optimization")
    parser.add_argument("--pdb_path", help="Path to PDB file (single input)")
    parser.add_argument("--ref_ligand_path", help="Path to reference ligand SDF file (optional for single input)")
    parser.add_argument("--pdb_metal_path", help="Path to PDB metal file (optional for single input)")
    parser.add_argument("--list_file", help="Path to list file (.pkl or .json) containing multiple inputs")
    parser.add_argument("--output_dir", default="output", help="Output directory for IFP files and JSON config")

    args = parser.parse_args()

    if not (args.pdb_path or args.list_file):
        parser.error("Either --pdb_path or --list_file must be provided")

    config_data = []
    if args.pdb_path:
        # Process single input
        config_item = process_single_input(
            args.pdb_path, args.ref_ligand_path, args.pdb_metal_path, args.output_dir
        )
        config_data = [config_item]
    elif args.list_file:
        # Process list input
        config_data = process_list_input(args.list_file, args.output_dir)

    if not config_data:
        logger.error("No valid configuration data generated")
        sys.exit(1)

    config_path = os.path.join(args.output_dir, 'IFPGen_config.json')
    save_config(config_data, config_path)