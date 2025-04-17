"""
Code for RNAbpFlow Flow Module supporting both training and inference
"""

import torch
import time
import os
import random
import numpy as np
import pandas as pd
import logging
from pytorch_lightning import LightningModule

from src.analysis import metrics
from src.analysis import utils as au
from src.models.flow_model import FlowModel
from src.models import utils as mu
from src.data.interpolant import Interpolant
from src.data.interpolant import Interpolant as InterpolantInf
from src.data import utils as du
from src.data import all_atom as rna_all_atom
from src.data import so3_utils
from src.data import nucleicacid
from src.data.nucleicacid import from_prediction, to_pdb
from pytorch_lightning.loggers.wandb import WandbLogger

class FlowModule(LightningModule):
    def __init__(self, cfg, folding_cfg=None):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._interpolant_cfg = cfg.interpolant
        
        # Set-up vector field prediction model
        self.model = FlowModel(cfg.model)
        
        # Set-up interpolant for training
        self.interpolant = Interpolant(cfg.interpolant)
        
        # Set-up interpolant for inference
        self.interpolant_inf = InterpolantInf(cfg.interpolant)
        
        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)
        
        self.save_hyperparameters()
    
    def on_train_start(self):
        self._epoch_start_time = time.time()
    
    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()
    
    def model_step(self, noisy_batch):
        """
        Params:
            noisy_batch (dict) : dictionary of tensors corresponding to corrupted Frame objects
            
        Remarks:
            Computes the different core and auxiliary losses between ground truth and predicted backbones
            
        Returns:
            Dictionary of core and auxiliary losses
        """
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask']
        is_na_residue_mask = noisy_batch["is_na_residue_mask"]
        
        if training_cfg.min_plddt_mask is not None:
            plddt_mask = noisy_batch['res_plddt'] > training_cfg.min_plddt_mask
            loss_mask *= plddt_mask
        
        num_batch, num_res = loss_mask.shape
        
        torsions_start_index = 0
        torsions_end_index = 9  # Using 9 angles instead of 8 for RNAbpFlow
        num_torsions = torsions_end_index - torsions_start_index
        
        if training_cfg.num_non_frame_atoms == 0:
            bb_filtered_atom_idx = [2, 3, 6]  # [C3', C4', O4']
        elif training_cfg.num_non_frame_atoms == 3:
            bb_filtered_atom_idx = [2, 3, 6] + [0, 7, 9]  # [C3', C4', O4'] + [C1', O3', P]
        elif training_cfg.num_non_frame_atoms == 7:
            bb_filtered_atom_idx = [2, 3, 6] + [0, 4, 7, 9, 10, 11, 12]  # [C3', C4', O4'] + [C1', C5', O3', P, OP1, OP2, N1]
        else:
            # NOTE: default is the original frame
            bb_filtered_atom_idx = [2, 3, 6]  # [C3', C4', O4']
        
        n_merged_atoms = len(bb_filtered_atom_idx)
        c1_idx = 0  # C1' atom index for base pair distance calculation
        
        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        gt_torsions_1 = noisy_batch['torsion_angles_sin_cos'][:, :, torsions_start_index:torsions_end_index, :].reshape(num_batch, num_res, num_torsions * 2)
        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, gt_rotmats_1.type(torch.float32))
        
        # Get secondary structure information
        ss = noisy_batch['ss']
        
        # Get one-hot encoding
        onehot = noisy_batch['onehot']
        
        gt_bb_atoms = rna_all_atom.to_atom37_rna(
            gt_trans_1, gt_rotmats_1,
            torch.ones_like(is_na_residue_mask),
            torsions=gt_torsions_1
        )
        gt_bb_atoms = gt_bb_atoms[:, :, bb_filtered_atom_idx]
        
        # Timestep used for normalization.
        t = noisy_batch['t']
        norm_scale = 1 - torch.min(t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        
        # Model output predictions - include onehot and ss in the input
        noisy_batch_with_ss = noisy_batch.copy()
        noisy_batch_with_ss['ss'] = ss
        noisy_batch_with_ss['onehot'] = onehot
        
        model_output = self.model(noisy_batch_with_ss)
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_torsions_1 = model_output['pred_torsions'].reshape(num_batch, num_res, num_torsions * 2)
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        
        # 2D base pair loss (bp2D)
        pair_feat = model_output['pair_feat']
        gt_pair_feat = ss
        # Using BCE with logits as mentioned in the paper
        # Equation 9: Lbp2D = −∑(SS⊙SS_hat - log(1+e^SS_hat))
        bp2d_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pair_feat, gt_pair_feat, reduction='mean'
        )
        
        # Backbone atom loss
        pred_bb_atoms = rna_all_atom.to_atom37_rna(
            pred_trans_1, pred_rotmats_1,
            torch.ones_like(is_na_residue_mask),
            torsions=pred_torsions_1
        )
        pred_bb_atoms = pred_bb_atoms[:, :, bb_filtered_atom_idx]
        
        gt_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / norm_scale[..., None]
        loss_denom = torch.sum(loss_mask, dim=-1) * n_merged_atoms
        bb_atom_loss = torch.sum((gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None], dim=(-1, -2, -3)) / loss_denom
        
        # Translation VF loss - Equation 5
        trans_error = (gt_trans_1 - pred_trans_1) / norm_scale * training_cfg.trans_scale
        loss_denom = torch.sum(loss_mask, dim=-1) * 3  # 3 frame atoms
        trans_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom
        
        # Rotation VF loss - Equation 6
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / norm_scale
        loss_denom = torch.sum(loss_mask, dim=-1) * 3  # 3 frame atoms
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom
        
        # 3D base pair distance loss (bp3D) - Equation 8
        # Extract base pair annotations from the three methods in ss
        # For each batch and base pair annotation method
        bp3d_loss = torch.zeros(num_batch, device=trans_loss.device)
        
        if 'base_pairs' in noisy_batch:
            base_pairs = noisy_batch['base_pairs']
            # Get C1' atom positions for both ground truth and predicted structures
            gt_all_atoms = rna_all_atom.to_atom37_rna(
                gt_trans_1, gt_rotmats_1,
                torch.ones_like(is_na_residue_mask),
                torsions=gt_torsions_1
            )
            pred_all_atoms = rna_all_atom.to_atom37_rna(
                pred_trans_1, pred_rotmats_1,
                torch.ones_like(is_na_residue_mask),
                torsions=pred_torsions_1
            )
            
            gt_c1_positions = gt_all_atoms[:, :, c1_idx]   # Get C1' positions
            pred_c1_positions = pred_all_atoms[:, :, c1_idx]  # Get C1' positions
            
            for b in range(num_batch):
                # Calculate distances for each base pair
                batch_bp_loss = 0
                total_bp_count = 0
                
                for bp_method_idx in range(3):  # For the three base pair annotation methods
                    if f'method_{bp_method_idx}' in base_pairs:
                        method_pairs = base_pairs[f'method_{bp_method_idx}'][b]
                        if len(method_pairs) > 0:
                            bp_count = len(method_pairs)
                            
                            # For each base pair, calculate the C1'-C1' distance difference
                            method_loss = 0
                            for m, n in method_pairs:
                                # Calculate ground truth and predicted distances
                                gt_dist = torch.norm(gt_c1_positions[b, m] - gt_c1_positions[b, n])
                                pred_dist = torch.norm(pred_c1_positions[b, m] - pred_c1_positions[b, n])
                                method_loss += (gt_dist - pred_dist) ** 2
                            
                            if bp_count > 0:
                                method_loss /= bp_count
                                batch_bp_loss += method_loss
                                total_bp_count += 1
                
                if total_bp_count > 0:
                    bp3d_loss[b] = batch_bp_loss / total_bp_count
        
        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res * n_merged_atoms, 3])
        gt_pair_dists = torch.linalg.norm(gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1)
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res * n_merged_atoms, 3])
        pred_pair_dists = torch.linalg.norm(pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)
        
        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, n_merged_atoms))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res * n_merged_atoms])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, n_merged_atoms))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res * n_merged_atoms])
        
        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]
        
        dist_mat_loss = torch.sum((gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask, dim=(1, 2))
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) - num_res)
        
        # Torsion angles loss - using 9 angles for RNAbpFlow (Equation 7)
        pred_torsions_1 = pred_torsions_1.reshape(num_batch, num_res, num_torsions, 2)
        gt_torsions_1 = gt_torsions_1.reshape(num_batch, num_res, num_torsions, 2)
        loss_denom = torch.sum(loss_mask, dim=-1) * num_torsions
        tors_loss = training_cfg.tors_loss_scale * torch.sum(
            torch.linalg.norm(pred_torsions_1 - gt_torsions_1, dim=-1) ** 2 * loss_mask[..., None], dim=(-1, -2)
        ) / loss_denom
        
        assert bb_atom_loss.shape[0] == dist_mat_loss.shape[0] == tors_loss.shape[0], f"Loss tensors shape mismatch: {bb_atom_loss.shape} vs {dist_mat_loss.shape} vs {tors_loss.shape}"
        
        # Combine losses according to Equation 10: Ltotal = 2 × Ltrans + Lrot + Ltors + Lbp3D + Lbp2D
        se3_vf_loss = 2 * trans_loss + rots_vf_loss
        
        # Including the base pair losses in the auxiliary loss
        auxiliary_loss = (bb_atom_loss + dist_mat_loss + tors_loss + bp3d_loss + bp2d_loss) * (t[:, 0] > training_cfg.aux_loss_t_pass)
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
        
        if torch.isnan(auxiliary_loss).any():
            print("NaN loss in aux_loss")
            auxiliary_loss = torch.zeros_like(auxiliary_loss).to(se3_vf_loss.device)
        
        if torch.isnan(se3_vf_loss).any():
            print("NaN loss in se3_vf_loss")
            se3_vf_loss = torch.zeros_like(se3_vf_loss).to(se3_vf_loss.device)
        
        return {
            "bb_atom_loss": bb_atom_loss,
            "trans_loss": trans_loss,
            "dist_mat_loss": dist_mat_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": se3_vf_loss,
            "torsion_loss": tors_loss,
            "bp3d_loss": bp3d_loss,
            "bp2d_loss": bp2d_loss
        }
    
    def training_step(self, batch, batch_idx):
        """
        Performs one iteration of SE(3) flow matching and returns total training loss
        using the core and auxiliary losses computed in `model_step`.
        """
        step_start_time = time.time()
        self.interpolant.set_device(batch['res_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)
        
        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                # Add ss and onehot to noisy_batch for RNAbpFlow
                noisy_batch_with_ss = noisy_batch.copy()
                noisy_batch_with_ss['ss'] = batch['ss']
                noisy_batch_with_ss['onehot'] = batch['onehot']
                
                model_sc = self.model(noisy_batch_with_ss)
                noisy_batch['trans_sc'] = model_sc['pred_trans']
        
        # Add ss and onehot to noisy_batch for model_step
        noisy_batch['ss'] = batch['ss']
        noisy_batch['onehot'] = batch['onehot']
        
        # Add base pair information if available
        if 'base_pairs' in batch:
            noisy_batch['base_pairs'] = batch['base_pairs']
        
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses['bb_atom_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k, v in batch_losses.items()
        }
        
        for k, v in total_losses.items():
            self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        # Losses to track. Stratified across t.
        t = torch.squeeze(noisy_batch['t'])
        self._log_scalar(
            "train/t",
            np.mean(du.to_numpy(t)),
            prog_bar=False, batch_size=num_batch)
        
        for loss_name, loss_dict in batch_losses.items():
            stratified_losses = mu.t_stratified_loss(
                t, loss_dict, loss_name=loss_name)
            for k, v in stratified_losses.items():
                self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        # Training throughput
        self._log_scalar("train/length", batch['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
        self._log_scalar("train/batch_size", num_batch, prog_bar=False)
        
        step_time = time.time() - step_start_time
        self._log_scalar("train/eps", num_batch / step_time)
        
        # Compute total loss according to Equation 10, ensuring all required components are included
        train_loss = (
            total_losses[self._exp_cfg.training.loss] +
            total_losses['auxiliary_loss']
        )
        self._log_scalar("train/loss", train_loss, batch_size=num_batch)
        
        return train_loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )
    
    def predict_step(self, batch, batch_idx=None):
        """
        Params:
            batch (dict) : dictionary containing RNA sequence and structure information
            
        Remarks:
            Used for conditional sampling from the Interpolant.
            Saves the generated all-atom backbones as PDB files at the specified saving directory.
        """
        
        # Use the original inference code from RNAbpFlow_flow_module_inf.py
        device = batch['sample_id'].device
        self.interpolant_inf.set_device(device)
        
        sample_id = batch['sample_id'].item()
        sample_dir = self._output_dir if hasattr(self, '_output_dir') else self._sample_write_dir
        
        ss = batch['ss']
        ss = ss.float()
        
        onehot = batch['onehot']
        onehot = onehot.float()
        
        seedval = batch['seed'] if 'seed' in batch else None
        
        map_dict = batch['mapfeat'] if 'mapfeat' in batch else None
        
        sample_length = batch['onehot'].shape[-2]
        
        sample = self.interpolant_inf.sample(
            1, 
            sample_length, 
            ss, 
            self.model, 
            seedval, 
            onehot, 
            map_dict, 
            self._interpolant_cfg.sampling.num_timesteps
        )
        
        results = {}
        results['final_atom_positions'] = sample
        
        indexlist = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0],
        ]
        
        feats = {}
        
        if map_dict is not None:
            feats['aatype'] = map_dict['aatype'].cpu().numpy()
        else:
            # Create aatype from onehot if mapfeat not available
            feats['aatype'] = torch.argmax(onehot, dim=-1).cpu().numpy()
        
        idx = np.arange(sample_length)
        idx = idx[None, :]
        feats['residue_index'] = idx
        
        typelist = list(feats['aatype'][0])
        
        final_atom_mask = []
        
        for i in range(len(typelist)):
            nttype = typelist[i]
            final_atom_mask.append(indexlist[nttype])
        
        final_atom_mask = np.array(final_atom_mask)
        
        results['final_atom_mask'] = np.expand_dims(final_atom_mask, axis=0)
        
        structure = from_prediction(feats, results)
        
        pdbpath = os.path.join(sample_dir, f'Sample_{sample_id}.pdb')
        
        with open(f"{pdbpath}", "wt") as f:
            print(to_pdb(structure), file=f)
            
        return pdbpath
    
    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )