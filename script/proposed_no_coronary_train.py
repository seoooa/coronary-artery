"""Train the proposed model with coronary-free seven-channel conditioning."""

from __future__ import annotations

import os
from pathlib import Path

import autorootcwd
import click
import lightning.pytorch as pytorch_lightning
import torch
from dvclive.lightning import DVCLiveLogger
from lightning.pytorch.callbacks import StochasticWeightAveraging
from monai.transforms import AsDiscrete, Compose, EnsureType

from script.proposed_train import CoronaryArterySegmentModel, print_monai_config
from src.data.proposed_no_coronary_dataloader import (
    NUM_SEG_CHANNELS,
    CoronaryArteryNoCoronaryDataModule,
)
from src.losses.losses import LossFactory
from src.metrics.metrics import MetricFactory
from src.models.proposed_networks import NetworkFactory


class CoronaryArteryNoCoronarySegmentModel(CoronaryArterySegmentModel):
    """Proposed model configured for background plus six anatomical classes."""

    def __init__(
        self,
        arch_name="UNETR",
        loss_fn="DiceFocalLoss",
        batch_size=1,
        lr=1e-3,
        patch_size=(96, 96, 96),
        label_nc=NUM_SEG_CHANNELS,
    ):
        # Initialize Lightning directly and reuse the base class's train/val/test
        # methods without first constructing an incompatible eight-channel model.
        pytorch_lightning.LightningModule.__init__(self)
        if label_nc != NUM_SEG_CHANNELS:
            raise ValueError(
                f"no-coronary training requires label_nc={NUM_SEG_CHANNELS}, "
                f"got {label_nc}"
            )

        self.loss_fn = loss_fn
        self.label_nc = label_nc
        self._model = NetworkFactory.create_network(
            arch_name, patch_size, label_nc=label_nc
        )
        self.loss_function = LossFactory.create_loss(loss_fn)
        self.metrics = MetricFactory.create_metrics()
        self.post_pred = Compose(
            [EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=2)]
        )
        self.post_label = Compose(
            [EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2)]
        )
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.validation_step_outputs = []
        self.batch_size = batch_size
        self.lr = lr
        self.patch_size = patch_size
        self.result_folder = Path("result")
        self.test_step_outputs = []


@click.command()
@click.option(
    "--arch_name",
    type=click.Choice(
        [
            "SegResNet",
            "UNETR",
            "SwinUNETR",
            "nnFormer",
            "CSNet3D",
            "AttentionUnet",
            "VNet",
        ]
    ),
    default="SegResNet",
    show_default=True,
)
@click.option(
    "--loss_fn",
    type=click.Choice(
        ["DiceLoss", "DiceCELoss", "DiceFocalLoss", "SoftDiceclDiceLoss"]
    ),
    default="DiceFocalLoss",
    show_default=True,
)
@click.option("--max_epochs", type=int, default=200, show_default=True)
@click.option("--check_val_every_n_epoch", type=int, default=10, show_default=True)
@click.option("--gpu_number", type=str, default="0", show_default=True)
@click.option("--checkpoint_path", type=click.Path(path_type=Path), default=None)
@click.option(
    "--guide",
    type=click.Choice(["segMap", "distanceMap"]),
    default="distanceMap",
    show_default=True,
)
@click.option(
    "--data_dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("data/imageCAS"),
    show_default=True,
)
@click.option(
    "--conditioning_dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("data/imageCAS_no_coronary_conditioning"),
    show_default=True,
)
def main(
    arch_name,
    loss_fn,
    max_epochs,
    check_val_every_n_epoch,
    gpu_number,
    checkpoint_path,
    guide,
    data_dir,
    conditioning_dir,
):
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.set_float32_matmul_precision("medium")
    print_monai_config()

    guide_name = "dstMap" if guide == "distanceMap" else "segMap"
    log_dir = Path(
        "result/experiments/no_coronary_conditioning"
    ) / f"proposed_{arch_name}_{guide_name}_{loss_fn}"
    log_dir.mkdir(parents=True, exist_ok=True)

    if "," in gpu_number:
        devices = [int(gpu) for gpu in gpu_number.split(",")]
        strategy = "ddp_find_unused_parameters_true"
    else:
        devices = [int(gpu_number)]
        strategy = "auto"

    callbacks = [
        StochasticWeightAveraging(
            swa_lrs=[1e-4], annealing_epochs=5, swa_epoch_start=100
        )
    ]
    trainer = pytorch_lightning.Trainer(
        devices=devices,
        strategy=strategy,
        max_epochs=max_epochs,
        logger=DVCLiveLogger(log_model=True, dir=str(log_dir), report="html"),
        enable_checkpointing=True,
        benchmark=True,
        accumulate_grad_batches=5,
        precision="bf16-mixed",
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        default_root_dir=str(log_dir),
    )

    data_module = CoronaryArteryNoCoronaryDataModule(
        data_dir=str(data_dir),
        conditioning_dir=str(conditioning_dir),
        batch_size=1,
        patch_size=(96, 96, 96),
        num_workers=4,
        cache_rate=0.05,
        use_distance_map=guide == "distanceMap",
    )
    data_module.prepare_data()

    model_kwargs = {
        "arch_name": arch_name,
        "loss_fn": loss_fn,
        "batch_size": 1,
        "label_nc": NUM_SEG_CHANNELS,
    }

    if checkpoint_path is not None:
        if not checkpoint_path.is_file():
            raise click.ClickException(f"checkpoint not found: {checkpoint_path}")

        checkpoint_filename = checkpoint_path.name
        if "final_model.ckpt" in checkpoint_filename:
            print(f"Loading no-coronary checkpoint for testing: {checkpoint_path}")
            model = CoronaryArteryNoCoronarySegmentModel.load_from_checkpoint(
                str(checkpoint_path), **model_kwargs
            )
            model.result_folder = log_dir
            trainer.test(model=model, datamodule=data_module)
            return

        if "epoch=" in checkpoint_filename:
            print(f"Resuming no-coronary training from: {checkpoint_path}")
            model = CoronaryArteryNoCoronarySegmentModel(**model_kwargs)
            model.result_folder = log_dir
            trainer.fit(model, datamodule=data_module, ckpt_path=str(checkpoint_path))
            trainer.save_checkpoint(str(log_dir / "final_model.ckpt"))
            trainer.test(model=model, datamodule=data_module)
            return

        print(f"Loading no-coronary checkpoint for testing: {checkpoint_path}")
        model = CoronaryArteryNoCoronarySegmentModel.load_from_checkpoint(
            str(checkpoint_path), **model_kwargs
        )
        model.result_folder = log_dir
        trainer.test(model=model, datamodule=data_module)
        return

    model = CoronaryArteryNoCoronarySegmentModel(**model_kwargs)
    model.result_folder = log_dir
    trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint(str(log_dir / "final_model.ckpt"))
    trainer.test(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
