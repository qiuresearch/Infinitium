import os
import GPUtil
import torch

import hydra
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
# from pytorch_lightning.trainer import Trainer # Duplicate import removed
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data.pdb_na_datamodule_base import PDBNABaseDataModule
from src.models.flow_module import FlowModule
import src.utils as eu
import wandb

log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')

class Experiment:
    def __init__(self, *, cfg: DictConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data_cfg
        self._exp_cfg = cfg.experiment
        self._model = FlowModule(self._cfg)
        self._datamodule = PDBNABaseDataModule(data_cfg=self._data_cfg)

    def train(self):
        callbacks = []

        if self._exp_cfg.debug:
            log.info("Debug mode.")
            logger = None
            self._exp_cfg.num_devices = 1
            self._data_cfg.loader.num_workers = 0
        else:
            logger = WandbLogger(**self._exp_cfg.wandb,)

            # Checkpoint directory
            ckpt_dir = self._exp_cfg.checkpointer.dirpath
            os.makedirs(ckpt_dir, exist_ok=True)
            log.info(f"Checkpoints saved to {ckpt_dir}")

            # Model checkpoints
            callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))

            # Save config
            cfg_path = os.path.join(ckpt_dir, 'config.yaml')
            with open(cfg_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f.name)
            cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
            flat_cfg = dict(eu.flatten_dict(cfg_dict))
            if isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
                logger.experiment.config.update(flat_cfg)

        devices = GPUtil.getAvailable(order='memory', limit = 8)[:self._exp_cfg.num_devices]
        log.info(f"Using devices: {devices}")

        trainer = Trainer(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=devices,
        )

        trainer.fit(
            model=self._model,
            datamodule=self._datamodule,
            ckpt_path=self._exp_cfg.warm_start
        )

@hydra.main(version_base=None, config_path="C:/Users/nikhi/Desktop/RNA/test/RNAbpFlow/configs", config_name="config")
def main(cfg: DictConfig):
    if cfg.experiment.warm_start is not None and cfg.experiment.warm_start_cfg_override:
 
        if os.path.isabs(cfg.experiment.warm_start):
             warm_start_cfg_path = os.path.join(os.path.dirname(cfg.experiment.warm_start), 'config.yaml')
        else:
           
             warm_start_cfg_path = os.path.join(os.path.dirname(cfg.experiment.warm_start), 'config.yaml')
            

        if os.path.exists(warm_start_cfg_path):
            warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

            # Warm start config may not have latest fields in the base config.
            # Add these fields to the warm start config.
            OmegaConf.set_struct(cfg.model, False)
            OmegaConf.set_struct(warm_start_cfg.model, False)
            # Merge loaded config into the main config under the 'model' key specifically
            cfg.model = OmegaConf.merge(cfg.model, warm_start_cfg.model)
            OmegaConf.set_struct(cfg.model, True)
            log.info(f'Loaded and merged warm start model config from {warm_start_cfg_path}')
        else:
             log.warning(f"Warm start config path specified but not found: {warm_start_cfg_path}")


    exp = Experiment(cfg=cfg)
    exp.train()

if __name__ == "__main__":
    main()