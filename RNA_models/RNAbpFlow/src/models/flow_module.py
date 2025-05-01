import torch
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger

from src.models.flow_model import FlowModel
from src.data.interpolant import Interpolant 
from src.data.nucleicacid import from_prediction, to_pdb
from src.analysis import metrics
from src.data import so3_utils
from src.data import utils as du
from src.data import base_constants


class FlowModule(LightningModule):
    def __init__(self, cfg, folding_cfg=None):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._interpolant_cfg = cfg.interpolant
        
        # Set-up vector field prediction model
        self.model = FlowModel(cfg.model)
        
        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)
        
        # Set up directories for samples and checkpoints
        self._sample_write_dir = self._exp_cfg.checkpointer.dirpath
        os.makedirs(self._sample_write_dir, exist_ok=True)
        
        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
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
            prog_bar=False,
            sync_dist=True
        )
        self._epoch_start_time = time.time()
    
    def model_step(self, noisy_batch):
        """
        Computes losses between predicted and ground truth structures
        
        Args:
            noisy_batch (dict): Batch of data with noise applied
            
        Returns:
            dict: Dictionary of losses
        """
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask']
        is_na_residue_mask = noisy_batch["is_na_residue_mask"]
        
        if training_cfg.min_plddt_mask is not None and 'res_plddt' in noisy_batch:
            plddt_mask = noisy_batch['res_plddt'] > training_cfg.min_plddt_mask
            loss_mask *= plddt_mask
        
        num_batch, num_res = loss_mask.shape
        
        # Ground truth labels
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        
        
        num_torsions = 9
        
        # Assumes torsion_angles_sin_cos shape is [batch, num_res, 18]
        
        gt_torsions_raw = noisy_batch['torsion_angles_sin_cos']
         

        gt_torsions_1 = gt_torsions_raw[:, :, :9, :]  
        

        # Optional flattening if needed
        # gt_torsions_1 = gt_torsions_1.reshape(num_batch, num_res, 18)




        
        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, gt_rotmats_1.type(torch.float32))
        
        # Timestep used for normalization
        t = noisy_batch['t']
        norm_scale = 1 - torch.min(t[..., None], torch.tensor(training_cfg.t_normalize_clip))
        
        # Model output predictions
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_torsions_1 = model_output['pred_torsions'].reshape(num_batch, num_res, num_torsions * 2)
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        
        # Translation VF loss
        trans_error = (gt_trans_1 - pred_trans_1) / norm_scale * training_cfg.trans_scale
        loss_denom = torch.sum(loss_mask, dim=-1) * 3  # 3 translational dimensions
        trans_loss = training_cfg.translation_loss_weight * torch.sum(
            trans_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom
        
        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / norm_scale
        loss_denom = torch.sum(loss_mask, dim=-1) * 3  # 3 rotational dimensions
        rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
            rots_vf_error ** 2 * loss_mask[..., None],
            dim=(-1, -2)
        ) / loss_denom
        
        # Torsion angles loss
        pred_torsions_1 = pred_torsions_1.reshape(num_batch, num_res, num_torsions, 2)
        gt_torsions_1 = gt_torsions_1.reshape(num_batch, num_res, num_torsions, 2)
        loss_denom = torch.sum(loss_mask, dim=-1) * num_torsions * 2  # num_torsions torsion angles × 2 components
        tors_loss = training_cfg.tors_loss_scale * torch.sum(
            torch.linalg.norm(pred_torsions_1 - gt_torsions_1, dim=-1) ** 2 * loss_mask[..., None], 
            dim=(-1, -2)
        ) / loss_denom
        
        # Base pair 3D loss (bp3D)
        bp3d_loss = torch.zeros_like(trans_loss)
        if 'bp_distances' in model_output and 'gt_bp_distances' in noisy_batch:
            gt_bp_distances = noisy_batch['gt_bp_distances']  # [batch, 3, num_bp_pairs]
            pred_bp_distances = model_output['bp_distances']  # [batch, 3, num_bp_pairs]
            
            # Calculate distance error for each base pair annotation method (3 methods)
            bp_distance_error = (gt_bp_distances - pred_bp_distances) ** 2
            
            # Average over the three annotation methods and all base pairs
            num_bp_methods = gt_bp_distances.shape[1]  # Should be 3
            
            # # Create mask for valid base pairs if available
            # if 'bp_mask' in noisy_batch:
            #     bp_mask = noisy_batch['bp_mask']  # [batch, 3, num_bp_pairs]
            #     bp3d_loss = training_cfg.bp3d_loss_weight * torch.sum(
            #         bp_distance_error * bp_mask, dim=(-1, -2)
            #     ) / (torch.sum(bp_mask, dim=(-1, -2)) + 1e-6)
            # else:
            # If no mask, assume all base pairs are valid
            # training_cfg.bp3d_loss_weight = 1
            bp3d_loss = training_cfg.bp3d_loss_weight * torch.mean(
                bp_distance_error, dim=(-1, -2)
            )
        
        # Base pair 2D loss (bp2D)
        bp2d_loss = torch.zeros_like(trans_loss)
        if 'pair_feat' in model_output and 'ss_bp' in noisy_batch:
            gt_ss_bp = noisy_batch['ss_bp'].float()  # [batch, 3, num_res, num_res]
            pred_ss_bp = model_output['pair_feat']  # [batch, 3, num_res, num_res]
            
            # Apply BCE loss as described in the paper
            # # Create mask for interaction pairs if available
            # if 'ss_bp_mask' in noisy_batch:
            #     ss_bp_mask = noisy_batch['ss_bp_mask']
                
            #     # Implementation of equation 9 from the paper
            #     bp2d_loss = 1 * torch.sum(
            #         -1 * (
            #             gt_ss_bp * torch.log(torch.sigmoid(pred_ss_bp) + 1e-10) +
            #             (1 - gt_ss_bp) * torch.log(1 - torch.sigmoid(pred_ss_bp) + 1e-10)
            #         ) * ss_bp_mask,
            #         dim=(-1, -2, -3)
            #     ) / (torch.sum(ss_bp_mask, dim=(-1, -2, -3)) + 1e-6)
            # else:
            # If no mask, use standard BCE
            bp2d_loss = 1 * torch.nn.functional.binary_cross_entropy_with_logits(
                pred_ss_bp, gt_ss_bp, reduction='mean'
            )
        
        # Secondary structure loss (if available in model output)
        # ss_loss = torch.zeros_like(trans_loss)
        # if 'pair_feat' in model_output and 'ss' in noisy_batch:
        #     gt_ss = noisy_batch['ss']
        #     pred_ss = model_output['pair_feat']
        #     ss_mask = loss_mask[:, :, None] * loss_mask[:, None, :]
        #     ss_error = (gt_ss - pred_ss) ** 2
        #     ss_loss = training_cfg.ss_loss_weight * torch.sum(
        #         ss_error * ss_mask[..., None],
        #         dim=(-1, -2, -3)
        #     ) / (torch.sum(ss_mask) * gt_ss.shape[-1])
        
        se3_vf_loss = trans_loss + rots_vf_loss
        
        # Combine base pair losses
        bp_loss = bp3d_loss + bp2d_loss
        
        # Combined auxiliary loss with torsion and secondary structure components
        auxiliary_loss = (tors_loss +  bp_loss) * (t[:, 0] > training_cfg.aux_loss_t_pass)
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
        
        if torch.isnan(auxiliary_loss).any():
            print("NaN loss in aux_loss")
            auxiliary_loss = torch.zeros_like(auxiliary_loss).to(se3_vf_loss.device)
        
        if torch.isnan(se3_vf_loss).any():
            print("NaN loss in se3_vf_loss")
            se3_vf_loss = torch.zeros_like(se3_vf_loss).to(se3_vf_loss.device)
        
        return {
            "trans_loss": trans_loss,
            "rots_vf_loss": rots_vf_loss,
            "se3_vf_loss": se3_vf_loss,
            "torsion_loss": tors_loss,
            #"ss_loss": ss_loss,
            "bp3d_loss": bp3d_loss,
            "bp2d_loss": bp2d_loss,
            "bp_loss": bp_loss,
            "auxiliary_loss": auxiliary_loss
        }
    
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
    
    def training_step(self, batch, batch_idx):
        """
        Performs one iteration of conditional SE(3) flow matching
        
        Args:
            batch (dict): Dictionary containing inputs and targets
            batch_idx (int): Batch index
            
        Returns:
            torch.Tensor: Total training loss
        """
        step_start_time = time.time()
        
        # Prepare inputs and targets
        #input_feats = {k: batch[k] for k in ['atom_map', 'aatype', 'onehot', 'ss', 'seed'] if k in batch}
        # target_feats = {
        #     k: batch[k] for k in [
        #         'torsion_angles_sin_cos', 'rotmats_1', 'trans_1', 'res_mask', 
        #         'is_na_residue_mask', 'ss_bp', 'gt_bp_distances', 'bp_mask', 'ss_bp_mask'
        #     ] if k in batch
        # }
        

        input_feats=batch['input_feats']
        target_feats=batch['target_feats']
        # Map ss → ss_bp for base-pair loss
        target_feats['ss_bp'] = input_feats['ss']
        # Create mask where ss_bp is non-zero
        target_feats['ss_bp_mask'] = (target_feats['ss_bp'].abs().sum(dim=-1, keepdim=True) > 0).float()


        # Combine for model input
        self.interpolant.set_device(target_feats['res_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch({**input_feats, **target_feats})
        

        
        # Apply self-conditioning if enabled
        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch['trans_sc'] = model_sc['pred_trans']
                if 'pair_feat' in model_sc:
                    noisy_batch['pair_feat_sc'] = model_sc['pair_feat']
        
        # Get losses
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses['trans_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k, v in batch_losses.items()
        }
        
        # Log losses
        for k, v in total_losses.items():
            self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Print loss values
       
        #print({k: v.item() for k, v in total_losses.items()})    
        

        # Stratify losses by timestep
        t = torch.squeeze(noisy_batch['t'])
        self._log_scalar(
            "train/t",
            np.mean(du.to_numpy(t)),
            prog_bar=False, batch_size=num_batch)
        
        # Log sequence length and batch size
        self._log_scalar("train/length", target_feats['res_mask'].shape[0], prog_bar=False, batch_size=num_batch)
        self._log_scalar("train/batch_size", num_batch, prog_bar=False)
        
        # Log throughput
        step_time = time.time() - step_start_time
        self._log_scalar("train/eps", num_batch / step_time)
        
        # Compute total loss according to equation 10 in the paper
        # Ltotal = 2 × Ltrans + Lrot + Ltors + Lbp3D + Lbp2D
        train_loss = (
            2 * total_losses['trans_loss'] +  # Translation loss with weight 2
            total_losses['rots_vf_loss'] +    # Rotation loss
            total_losses['auxiliary_loss']     # Contains torsion, bp3D, bp2D losses
        )
        self._log_scalar("train/loss", train_loss, batch_size=num_batch)
       

        
        return train_loss


    def validation_step(self, batch, batch_idx):
        """
        Generates samples and computes validation metrics
        
        Args:
            batch (dict): Dictionary containing inputs and targets
            batch_idx (int): Batch index
        """
        # Extract inputs
        #input_feats = {k: batch[k] for k in ['atom_map', 'aatype', 'onehot', 'ss', 'seed'] if k in batch}
        #target_feats = {k: batch[k] for k in ['torsion_angles_sin_cos', 'rotmats_1', 'trans_1', 'res_mask', 'is_na_residue_mask'] if k in batch}
        input_feats=batch["input_feats"]
        target_feats=batch["target_feats"]
        res_mask = target_feats['res_mask']
        is_na_residue_mask = target_feats['is_na_residue_mask'].bool()
        
        self.interpolant.set_device(res_mask.device)
        num_batch = 1  # Generate one sample per batch item
        num_res = is_na_residue_mask.sum(dim=-1).max().item()
        
        # Create sample
        ss = input_feats['ss'].float()
        onehot = input_feats['onehot'].float() if isinstance(input_feats['onehot'], torch.Tensor) else torch.tensor(input_feats['onehot']).float()
        seedval = input_feats['seed']
        #map_dict = {'aatype': input_feats['aatype']} if 'aatype' in input_feats else None
        map_dict = {'aatype': input_feats['aatype'],'atom_map': input_feats['atom_map']} if 'aatype' in input_feats and 'atom_map' in input_feats else None

        # Generate conditional sample
        sample = self.interpolant.sample(
            num_batch, 
            num_res, 
            ss, 
            self.model, 
            seedval, 
            onehot, 
            map_dict, 
            self._interpolant_cfg.sampling.num_timesteps
        )
        
        batch_metrics = []
        
        # Save sample and compute metrics
        for i in range(num_batch):
            final_pos = sample[i] if isinstance(sample, list) else sample
            
            if not isinstance(final_pos, np.ndarray):
                final_pos = final_pos.detach().cpu().numpy()
                
            # Save RNA atoms to PDB
            saved_rna_path = os.path.join(
                self._sample_write_dir, 
                f'valid_sample_{i}_idx_{batch_idx}_len_{num_res}.pdb'
            )
            
            
            # Create PDB using nucleicacid module
            results = {}
            results['final_atom_positions'] = final_pos
            
            # Create atom mask
            indexlist = [
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0],
            ]
            
            feats = {}
            aatype = input_feats['aatype'].cpu().numpy()
            feats['aatype'] = aatype
            
            idx = np.arange(aatype.shape[1])
            idx = idx[None, :]
            feats['residue_index'] = idx
            
            typelist = list(feats['aatype'][0] if feats['aatype'].ndim > 1 else feats['aatype'])
            final_atom_mask = []
            
            for i in range(len(typelist)):
                nttype = typelist[i]
                final_atom_mask.append(indexlist[nttype])
            
            final_atom_mask = np.array(final_atom_mask)
            results['final_atom_mask'] = np.expand_dims(final_atom_mask, axis=0)
           
            
            structure = from_prediction(feats, results)


            with open(saved_rna_path, "wt") as f:
                print(to_pdb(structure), file=f)
            
            # Log sample to W&B
            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_samples.append(
                    [saved_rna_path, self.global_step, wandb.Molecule(saved_rna_path)]
                )
            
            # Compute metrics
        metrics_dict = {}  # This replaces the current printing block

        try:
            c4_idx = base_constants.atom_order["C4\'"]
            c4_positions = structure.atom_positions[:, c4_idx]
            c4_mask = structure.atom_mask[:, c4_idx]
            valid_c4_positions = c4_positions[c4_mask > 0]
            rna_c4_c4_metrics = metrics.calc_rna_c4_c4_metrics(valid_c4_positions)

            metrics_dict.update(rna_c4_c4_metrics)

        except Exception as e:
            self._print_logger.error(f"Error computing metrics: {e}")
            metrics_dict = {}

        self.validation_epoch_metrics.append(metrics_dict)


    
    def on_validation_epoch_end(self):
        
        if len(self.validation_epoch_metrics) > 0:
            # Average metrics across all batches
            keys = self.validation_epoch_metrics[0].keys()
            avg_metrics = {
                k: sum(d.get(k, 0) for d in self.validation_epoch_metrics) / len(self.validation_epoch_metrics)
                for k in keys
            }

            print("Validation Metrics (Epoch Average):")
            for k, v in avg_metrics.items():
                print(f"{k}: {float(v):.4f}")

            print("------------------------------")

            self.validation_epoch_metrics.clear()

    
    def configure_optimizers(self):
        """Configure optimizer for the model"""
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )
    
    def predict_step(self, batch, batch_idx=None):
        """Generate sample during inference"""
        device = batch['onehot'].device if hasattr(batch['onehot'], 'device') else f'cuda:{torch.cuda.current_device()}'
        self.interpolant.set_device(device)
        
        sample_id = batch['sample_id'].item() if 'sample_id' in batch else 0
        sample_dir = getattr(self, '_output_dir', self._sample_write_dir)
        os.makedirs(sample_dir, exist_ok=True)
        
        ss = batch['ss'].float()
        onehot = batch['onehot'].float() if isinstance(batch['onehot'], torch.Tensor) else torch.tensor(batch['onehot'], device=device).float()
        seedval = batch['seed'] if 'seed' in batch else 42
        #map_dict = batch.get('mapfeat', {'aatype': batch['aatype']}) if 'aatype' in batch else None
        map_dict = batch['mapfeat']
        sample_length = onehot.shape[-2]
        num_timesteps = getattr(self._interpolant_cfg.sampling, 'num_timesteps', 50) if hasattr(self, '_interpolant_cfg') else 50
        
        sample = self.interpolant.sample(1, sample_length, ss, self.model, seedval, onehot, map_dict, num_timesteps)
        
        results = {}
        results['final_atom_positions'] = sample
        
        indexlist = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0],
        ]
        
        feats = {}
        
        aatype = map_dict['aatype'].cpu().numpy() if isinstance(map_dict, dict) and 'aatype' in map_dict else None
        if aatype is None and 'aatype' in batch:
            aatype = batch['aatype'].cpu().numpy()
        
        feats['aatype'] = aatype
        
        idx = np.arange(sample_length)
        idx = idx[None,:]
        feats['residue_index'] = idx
        
        typelist = list(feats['aatype'][0] if feats['aatype'].ndim > 1 else feats['aatype'])
        
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