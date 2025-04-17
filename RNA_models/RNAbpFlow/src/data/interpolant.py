import torch, copy
from src.data import so3_utils
from src.data import utils as du
from src.data import all_atom as rna_all_atom
import numpy as np

from src.data.tensor_utils import batched_gather

from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment

def _centered_gaussian(num_batch, num_res, device, seedval=-1):
    # returns randomly-sampled zero-centered Gaussian noise for translations
    if seedval != -1:
        torch.manual_seed(seedval)
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def _uniform_so3(num_batch, num_res, device, seedval=-1):
    # returns uniformly-sampled rotations (from U(SO(3)))
    if seedval != -1:
        np.random.seed(seedval)
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(), # random rotation
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    """Apply diffusion mask to translations"""
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])

def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    """Apply diffusion mask to rotations"""
    return (
        rotmats_t * diffuse_mask[..., None, None]
        + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )

class Interpolant:
    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        """
        Sample random timestep for flow matching
        From paper: "t is sampled from U([0, 1 − ϵ]) where ϵ = 0.1 for training stability"
        """
        min_t = 0.1  # ϵ = 0.1 as per paper
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - min_t)
    
    def _batch_ot(self, trans_0, trans_1, res_mask):
        """
        Use optimal transport to create the unified "flow" for 
        translations by computing a cost matrix
        """
        num_batch, num_res = trans_0.shape[:2]
        noise_idx, gt_idx = torch.where(
            torch.ones(num_batch, num_batch))
        batch_nm_0 = trans_0[noise_idx]
        batch_nm_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]
        aligned_nm_0, aligned_nm_1, _ = du.batch_align_structures(
            batch_nm_0, batch_nm_1, mask=batch_mask
        ) 
        aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
        aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)
        
        # Compute cost matrix of aligned noise to ground truth
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
        cost_matrix = torch.sum(
            torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
        ) / torch.sum(batch_mask, dim=-1)
        noise_perm, gt_perm = linear_sum_assignment(du.to_numpy(cost_matrix))
        return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]

    def _corrupt_trans(self, trans_1, t, res_mask):
        """
        Returns optimal transport-based interpolated "flow" for translations
        Based on paper equation: xt = (1 − t)x0 + tx1
        """
        # Sample from unit Gaussian as per paper
        trans_nm_0 = _centered_gaussian(*res_mask.shape, self._device)
        trans_0 = trans_nm_0 * du.NM_TO_ANG_SCALE
        
        # Apply optimal transport to create better initial flow
        trans_0 = self._batch_ot(trans_0, trans_1, res_mask)
        
        # Create geodesic flow for translations: xt = (1 − t)x0 + tx1
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        
        # Apply mask to keep valid residues
        trans_t = _trans_diffuse_mask(trans_t, trans_1, res_mask)
        return trans_t * res_mask[..., None]
    
    def _corrupt_rotmats(self, rotmats_1, t, res_mask):
        """
        Returns interpolated "flow" for rotations based on the paper
        Using IGSO3 distribution with σ = 1.5 as mentioned in the paper
        """
        num_batch, num_res = res_mask.shape
        
        # Sample from IGSO3(σ = 1.5) as per paper
        noisy_rotmats = self.igso3.sample(
            torch.tensor([1.5]),
            num_batch*num_res
        ).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        
        # Apply noise to ground truth rotations
        rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        
        # Implement rt = expr0(t · logr0(r1)) for SO(3) geodesic interpolation
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        
        # Apply mask
        identity = torch.eye(3, device=self._device)
        rotmats_t = (
            rotmats_t * res_mask[..., None, None]
            + identity[None, None] * (1 - res_mask[..., None, None])
        )
        
        return _rots_diffuse_mask(rotmats_t, rotmats_1, res_mask)

    def corrupt_batch(self, batch):
        """
        Params:
            batch (dict) : a dictionary where each key is a tensor name and each value is the corresponding tensor of Rigid objects.

        Returns:
            batch of corrupted/noisy/interpolated tensors for training
        """
        noisy_batch = copy.deepcopy(batch)

        trans_1 = batch['trans_1']  # [B, N, 3] (in Angstrom)
        rotmats_1 = batch['rotmats_1']  # [B, N, 3, 3]
        res_mask = batch['res_mask']  # [B, N]
        num_batch, _ = res_mask.shape

        # Sample random timestep t ~ U([0, 1-ϵ])
        t = self.sample_t(num_batch)[:, None]  # [B, 1]
        noisy_batch['t'] = t

        # Apply corruptions to translations
        trans_t = self._corrupt_trans(trans_1, t, res_mask)
        noisy_batch['trans_t'] = trans_t

        # Apply corruptions to rotations
        rotmats_t = self._corrupt_rotmats(rotmats_1, t, res_mask)
        noisy_batch['rotmats_t'] = rotmats_t
        
        # If we have sequence information, keep it
        if 'ss' in batch:
            noisy_batch['ss'] = batch['ss']
        
        if 'onehot' in batch:
            noisy_batch['onehot'] = batch['onehot']
            
        return noisy_batch

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        """
        Perform Euler ODESolve step for translations
        Based on paper equation: xt = xt−1 + (Δt/(1−t)) · (xt − xt−1)
        """
        trans_vf = (trans_1 - trans_t) / (1 - t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        """
        Perform Euler ODESolve step for rotations
        Based on paper equation: rt = exprt−1(c·Δt·logrt−1(rt))
        where c = 10 for better performance during inference
        """
        # During inference, use exponential scheduler with c = 10 as per paper
        c = 10  # scaling factor from paper
        
        # Implement the rotation update step
        return so3_utils.geodesic_t(c * d_t, rotmats_1, rotmats_t)

    def sample(self, num_batch, num_res, ss, model, seedval=-1, onehot=None, map_dict=None, num_timesteps=100):
        """
        Params:
            num_batch : number of independent samples
            num_res : number of nucleotides to model
            ss : secondary structure information
            model : the parameterized vector field (ie, the "denoiser")
            seedval : random seed for reproducibility
            onehot : one-hot encoding of RNA sequence
            map_dict : mapping dictionary for atom indices
            num_timesteps : number of steps for ODE integration

        Returns:
            Generated full atom RNA 3D structures
        """
        res_mask = torch.ones(num_batch, num_res, device=self._device)
        
        # Set-up initial prior samples
        trans_0 = _centered_gaussian(num_batch, num_res, self._device, seedval) * du.NM_TO_ANG_SCALE
        rotmats_0 = _uniform_so3(num_batch, num_res, self._device, seedval)
        
        batch = {
            'res_mask': res_mask,
        }

        # Get diffusion timesteps in order between [0, 1]
        # Starting from 1e-2 for numerical stability
        ts = torch.linspace(1e-2, 1.0, num_timesteps)
        t_1 = ts[0]

        prev_trans = trans_0
        prev_rot = rotmats_0

        for t_2 in ts[1:]:
            # Run one step of reverse diffusion (denoising)
            trans_t_1, rotmats_t_1 = prev_trans, prev_rot

            batch['trans_t'] = trans_t_1
            batch['rotmats_t'] = rotmats_t_1
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['t'] = t
            
            # Add conditional information
            batch['ss'] = ss
            if onehot is not None:
                batch['onehot'] = onehot
            
            with torch.no_grad():
                model_out = model(batch)

            # Store model outputs
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            
            # Self-conditioning (optional)
            batch['trans_sc'] = pred_trans_1

            # Take reverse step using Euler integration
            d_t = t_2 - t_1
            
            # Update translations and rotations using the flows from the paper
            trans_t_2 = self._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self._rots_euler_step(d_t, t_1, pred_rotmats_1, rotmats_t_1)
            
            prev_trans = trans_t_2
            prev_rot = rotmats_t_2
            t_1 = t_2

        # Final step at t=1
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prev_trans, prev_rot
        batch['trans_t'] = trans_t_1
        batch['rotmats_t'] = rotmats_t_1
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        
        with torch.no_grad():
            model_out = model(batch)  # Final predicted frames
        
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        pred_torsions_1 = model_out['pred_torsions']
        
        is_na_residue_mask = torch.ones(num_batch, num_res, device=self._device).bool()
        assert res_mask.shape == is_na_residue_mask.shape, "Shape mismatch between NA masks"

        # Convert to atom representation for output
        if map_dict is not None:
            pred_bb_atoms_23, _ = rna_all_atom.to_atom23_rna(
                                pred_trans_1, pred_rotmats_1, 
                                map_dict['aatype'],
                                torsions=pred_torsions_1,
                            )

            atom28_data = batched_gather(
                pred_bb_atoms_23,
                map_dict["atom_map"],
                dim=-2,
                no_batch_dims=len(pred_bb_atoms_23.shape[:-2]),
            )
            
            return atom28_data.detach().cpu().numpy()
        else:
            # Return frames if map_dict is not provided
            return pred_trans_1.detach().cpu().numpy(), pred_rotmats_1.detach().cpu().numpy()