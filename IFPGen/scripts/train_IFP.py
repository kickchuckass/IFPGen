import argparse
import os
import shutil

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

# Add parent directory to sys.path for custom imports
import sys
sys.path.append('..')

import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from datasets.IFP_dataset import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import IFPDiff


def get_auroc(y_true, y_pred, feat_mode: str) -> float:
    """
    Calculate average AUROC for atom types.

    Args:
        y_true: True labels.
        y_pred: Predicted probabilities.
        feat_mode: Ligand atom feature mode.

    Returns:
        Average AUROC score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.0
    possible_classes = set(y_true)
    mapping = {
        'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
        'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
        'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
    }
    atom_map = mapping[feat_mode]
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        print(f'atom: {atom_map[c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


def load_checkpoint(ckpt_path: str, model, optimizer, scheduler):
    """
    Load model checkpoint.

    Args:
        ckpt_path: Path to checkpoint file.
        model: Model instance.
        optimizer: Optimizer instance.
        scheduler: Scheduler instance.

    Returns:
        Loaded model, optimizer, scheduler, config, and starting iteration.
    """
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    config = checkpoint['config']
    iteration = checkpoint['iteration']
    return model, optimizer, scheduler, config, iteration


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train IFP Diffusion Model')
    parser.add_argument('config', type=str, help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--logdir', type=str, default='./logs_diffusion', help='Log directory')
    parser.add_argument('--tag', type=str, default='', help='Experiment tag')
    parser.add_argument('--train_report_iter', type=int, default=200, help='Training report interval')
    parser.add_argument('--index', type=str, default='../example/train/index_.pkl', help='Dataset index file')
    parser.add_argument('--processed', type=str, default='../example/processed', help='Processed dataset path')
    parser.add_argument('--split', type=str, default='../example/train/split_.pt', help='Dataset split path')
    args = parser.parse_args()

    # Load config
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    # Logging setup
    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    # writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('../models', os.path.join(log_dir, 'models'))

    # Data transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [protein_featurizer, ligand_featurizer, trans.FeaturizeLigandBond()]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Load dataset
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform,
        processed_path=args.processed,
        split=args.split,
        index=args.index
    )
    train_set, val_set = subsets['train'], subsets['val']
    logger.info(f'Training: {len(train_set)} | Validation: {len(val_set)}')

    collate_exclude_keys = [
        'ligand_nbh_list', 'ligand_rdmol', 'protein_atom2residue',
        'protein_filename', 'ligand_filename', 'ligand_res_idname_4_ligand',
        'ligand_bond_lengths', 'ligand_bond_angles', 'ligand_bond_angles_triplet'
    ]
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ))
    val_loader = DataLoader(
        val_set,
        config.train.batch_size,
        shuffle=False,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    )

    # Model setup
    logger.info(f'Log dir: {log_dir}')
    logger.info('Building model...')
    model = IFPDiff(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).to(args.device)
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)

    def train(it: int):
        """Perform a training iteration."""
        model.train()
        optimizer.zero_grad()
        for _ in range(config.train.n_acc_batch):
            batch = next(train_iterator).to(args.device)
            protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
            gt_protein_pos = batch.protein_pos + protein_noise

            results = model.get_diffusion_loss(
                protein_pos=gt_protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch.protein_element_batch,
                ligand_pos=batch.ligand_pos,
                ligand_v=batch.ligand_atom_feature_full,
                batch_ligand=batch.ligand_element_batch,
                ifp_info=batch.ifp_info.to(torch.float32),
            )
            loss = results['loss'] / config.train.n_acc_batch
            loss.backward()

        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        if it % args.train_report_iter == 0:
            logger.info(
                '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
                    it, results['loss'], results['loss_pos'], results['loss_v'],
                    optimizer.param_groups[0]['lr'], orig_grad_norm
                )
            )
            # for k, v in results.items():
            #     if torch.is_tensor(v) and v.squeeze().ndim == 0:
            #         writer.add_scalar(f'train/{k}', v, it)
            # writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            # writer.add_scalar('train/grad', orig_grad_norm, it)
            # writer.flush()

    def validate(it: int) -> float:
        """Perform validation."""
        sum_loss, sum_loss_pos, sum_loss_v, sum_n = 0, 0, 0, 0
        all_pred_v, all_true_v = [], []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate', miniters=100, mininterval=10):
                batch = batch.to(args.device)
                batch_size = batch.num_graphs
                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch_size).to(args.device)
                    results = model.get_diffusion_loss(
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        batch_protein=batch.protein_element_batch,
                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        batch_ligand=batch.ligand_element_batch,
                        time_step=time_step,
                        ifp_info=batch.ifp_info.to(torch.float32),
                    )
                    sum_loss += float(results['loss']) * batch_size
                    sum_loss_pos += float(results['loss_pos']) * batch_size
                    sum_loss_v += float(results['loss_v']) * batch_size
                    sum_n += batch_size
                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())

        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        atom_auroc = get_auroc(
            np.concatenate(all_true_v),
            np.concatenate(all_pred_v, axis=0),
            feat_mode=config.data.transform.ligand_atom_mode
        )

        if config.train.scheduler.type in ['plateau', 'warmup_plateau']:
            scheduler.step(avg_loss) if config.train.scheduler.type == 'warmup_plateau' else scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info(
            '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Avg atom auroc %.6f' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000, atom_auroc
            )
        )
        # writer.add_scalar('val/loss', avg_loss, it)
        # writer.add_scalar('val/loss_pos', avg_loss_pos, it)
        # writer.add_scalar('val/loss_v', avg_loss_v, it)
        # writer.flush()
        return avg_loss

    # Training loop
    start_iteration = 1
    if config.model.ckpt_path != '':
        print("config.model.ckpt_path: ",config.model.ckpt_path)
        model, optimizer, scheduler, config, start_iteration = load_checkpoint(
            config.model.ckpt_path, model, optimizer, scheduler
        )
        logger.info(f'Checkpoint loaded. Resuming from iteration {start_iteration}')
        logger.info(f'Loaded config: {config}')

    best_loss, best_iter = None, None
    try:
        for it in range(start_iteration, config.train.max_iters + 1):
            try:
                train(it)
                if it % config.train.val_freq == 0 or it == config.train.max_iters:
                    val_loss = validate(it)
                    if best_loss is None or val_loss < best_loss:
                        logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                        best_loss, best_iter = val_loss, it
                        ckpt_path = os.path.join(ckpt_dir, f'{it}.pt')
                        torch.save({
                            'config': config,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'iteration': it,
                        }, ckpt_path)
                    else:
                        logger.info(
                            f'[Validate] Val loss not improved. Best: {best_loss:.6f} at iter {best_iter}'
                        )
            except KeyboardInterrupt:
                logger.info('Skipping iteration due to interrupt')
                continue
    except KeyboardInterrupt:
        logger.info('Terminating...')
