import autorootcwd
import lightning.pytorch as pytorch_lightning
from monai.utils import set_determinism
from monai.transforms import (
    Compose,
    EnsureType,
    AsDiscrete,
    Resize,
)

from lightning.pytorch.callbacks import (
    BatchSizeFinder,
    LearningRateFinder,
    StochasticWeightAveraging,
)

from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.config import print_config
import torch
import os
import click
import numpy as np
import nibabel as nib
from pathlib import Path
import csv
from dvclive.lightning import DVCLiveLogger

from src.data.ape_dataloader import CoronaryArteryDataModule
from src.models.ape_networks import NetworkFactory
from src.losses.losses import LossFactory
from src.metrics.metrics import MetricFactory

def print_monai_config():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        print_config()

class CoronaryArterySegmentModel(pytorch_lightning.LightningModule):
    """Proposed model"""

    def __init__(
        self,
        arch_name="UNETR",
        loss_fn="DiceFocalLoss",
        batch_size=1,
        lr=1e-3,
        patch_size=(96, 96, 96),
    ):
        super().__init__()

        self._model = NetworkFactory.create_network(arch_name, patch_size)
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
        self.result_folder = Path("result")  # Define the result folder
        self.test_step_outputs = []

    def forward(self, x, seg, pos):
        # Modify the forward method to include seg, pos
        return self._model(x, seg, pos)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels, segs, poses = (
            batch["image"],
            batch["label"],
            batch["seg"],
            batch["pos"]
        )  # Include seg, pos
        output = self.forward(images, segs, poses)  # Pass seg, pos to forward
        loss = self.loss_function(output, labels)
        metrics = loss.item()
        self.log(
            "train_loss",
            metrics,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )   
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, segs, poses = (
            batch["image"],
            batch["label"],
            batch["seg"],
            batch["pos"]
        )  # Include seg, pos

        inputs = torch.cat((images, segs, poses), dim=1)

        roi_size = self.patch_size
        sw_batch_size = 4
        outputs = sliding_window_inference(
            inputs,
            roi_size,
            sw_batch_size,
            lambda x: self.forward(
                x[:, :1, ...], x[:, 1:2, ...], x[:, 2:, ...]
            ),  # Split before forward
        )

        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]

        MetricFactory.calculate_metrics(self.metrics, outputs, labels)
        
        d = {"val_loss": loss, "val_number": len(outputs)}
        self.validation_step_outputs.append(d)
        return d

    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]

        metric_results = MetricFactory.aggregate_metrics(self.metrics)
        MetricFactory.reset_metrics(self.metrics)

        mean_val_loss = torch.tensor(val_loss / num_items, device=self.device)
        log_dict = {
            "val_dice": torch.tensor(metric_results["dice"], device=self.device),
            "val_hausdorff": torch.tensor(metric_results["hausdorff"], device=self.device),
            "val_iou": torch.tensor(metric_results["iou"], device=self.device),
            "val_precision": torch.tensor(metric_results["precision"], device=self.device),
            "val_recall": torch.tensor(metric_results["recall"], device=self.device),
            "val_cldice": torch.tensor(metric_results["cldice"], device=self.device),
            "val_betti_0": torch.tensor(metric_results["betti_0"], device=self.device),
            "val_betti_1": torch.tensor(metric_results["betti_1"], device=self.device),
            "val_loss": mean_val_loss,
        }

        self.log_dict(log_dict, sync_dist=True)

        if metric_results["dice"] > self.best_val_dice:
            self.best_val_dice = metric_results["dice"]
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {metric_results['dice']:.4f}, "
            f"hausdorff: {metric_results['hausdorff']:.4f}, "
            f"iou: {metric_results['iou']:.4f}, "
            f"precision: {metric_results['precision']:.4f}, "
            f"recall: {metric_results['recall']:.4f}, "
            f"cldice: {metric_results['cldice']:.4f}, "
            f"betti_0: {metric_results['betti_0']:.4f}, "
            f"betti_1: {metric_results['betti_1']:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.validation_step_outputs.clear()  # free memory

    def save_result(self, inputs, outputs, labels, filename_prefix="result"):
        # Ensure outputs and labels are numpy arrays
        save_folder = self.result_folder / "test"
        os.makedirs(save_folder, exist_ok=True)

        inputs_np = inputs.detach().cpu().numpy().squeeze()
        outputs_np = outputs.detach().cpu().numpy().squeeze()[1]
        labels_np = labels.detach().cpu().numpy().squeeze()[1]

        # inputs_np = np.moveaxis(inputs_np, 0, -1)
        # outputs_np = np.moveaxis(outputs_np, 0, -1)
        # labels_np = np.moveaxis(labels_np, 0, -1)

        # Save inputs as NIfTI
        inputs_nifti = nib.Nifti1Image(
            inputs_np,
            np.array([[0.35, 0, 0, 0], [0, 0.35, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 1]]),
        )
        nib.save(inputs_nifti, save_folder / f"{filename_prefix}_inputs.nii.gz")

        # Save outputs as NIfTI
        outputs_nifti = nib.Nifti1Image(
            outputs_np,
            np.array([[0.35, 0, 0, 0], [0, 0.35, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 1]]),
        )
        nib.save(outputs_nifti, save_folder / f"{filename_prefix}_outputs.nii.gz")

        # Save labels as NIfTI
        labels_nifti = nib.Nifti1Image(
            labels_np,
            np.array([[0.35, 0, 0, 0], [0, 0.35, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 1]]),
        )
        nib.save(labels_nifti, save_folder / f"{filename_prefix}_labels.nii.gz")

        print(f"Result saved to: {save_folder}")
        print(f"Inputs: {save_folder / f'{filename_prefix}_inputs.nii.gz'}")
        print(f"Outputs: {save_folder / f'{filename_prefix}_outputs.nii.gz'}")
        print(f"Labels: {save_folder / f'{filename_prefix}_labels.nii.gz'}")

    def test_step(self, batch, batch_idx):
        images, labels, segs, poses = (
            batch["image"],
            batch["label"],
            batch["seg"],
            batch["pos"]
        )  # Include seg, pos
        roi_size = self.patch_size
        sw_batch_size = 4
        # Concatenate images and segs along the channel dimension
        inputs = torch.cat((images, segs, poses), dim=1)

        outputs = sliding_window_inference(
            inputs,
            roi_size,
            sw_batch_size,
            lambda x: self.forward(
                x[:, :1, ...], x[:, 1:2, ...], x[:, 2:, ...]
            ),  # Split before forward
        )

        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]

        filename = batch["image"].meta["filename_or_obj"][0]
        patient_id = filename.split("/")[-2]  # Gets patient id (ex. 25) from the path

        # Save result
        self.save_result(
            images, outputs[0], labels[0], filename_prefix=f"Subj_{patient_id}"
        )

        MetricFactory.calculate_metrics(self.metrics, outputs, labels)
        
        metric_results = MetricFactory.aggregate_metrics(self.metrics)
        
        d = {
            "test_dice": metric_results["dice"],
            "test_hausdorff": metric_results["hausdorff"],
            "test_iou": metric_results["iou"],
            "test_precision": metric_results["precision"],
            "test_recall": metric_results["recall"],
            "test_cldice": metric_results["cldice"],
            "test_betti_0": metric_results["betti_0"],
            "test_betti_1": metric_results["betti_1"],
            "patient_id": patient_id,
        }
        self.test_step_outputs.append(d)

        MetricFactory.reset_metrics(self.metrics)

        return d

    def on_test_epoch_end(self):
        # Calculate mean metrics
        dice_scores = [x["test_dice"] for x in self.test_step_outputs]
        hausdorff_scores = [x["test_hausdorff"] for x in self.test_step_outputs]
        iou_scores = [x["test_iou"] for x in self.test_step_outputs]
        precision_scores = [x["test_precision"] for x in self.test_step_outputs]
        recall_scores = [x["test_recall"] for x in self.test_step_outputs]
        cldice_scores = [x["test_cldice"] for x in self.test_step_outputs]
        betti_0_scores = [x["test_betti_0"] for x in self.test_step_outputs]
        betti_1_scores = [x["test_betti_1"] for x in self.test_step_outputs]

        # Calculate means
        mean_dice = np.mean(dice_scores)
        mean_hausdorff = np.mean(hausdorff_scores)
        mean_iou = np.mean(iou_scores)
        mean_precision = np.mean(precision_scores)
        mean_recall = np.mean(recall_scores)
        mean_cldice = np.mean(cldice_scores)
        mean_betti_0 = np.mean(betti_0_scores)
        mean_betti_1 = np.mean(betti_1_scores)

        # Log mean metrics
        self.log_dict({
            "test/mean_dice": mean_dice,
            "test/mean_hausdorff": mean_hausdorff,
            "test/mean_iou": mean_iou,
            "test/mean_precision": mean_precision,
            "test/mean_recall": mean_recall,
            "test/mean_cldice": mean_cldice,
            "test/mean_betti_0": mean_betti_0,
            "test/mean_betti_1": mean_betti_1,
        })

        # Save detailed result to CSV
        result_file = self.result_folder / "test" / "test_result.csv"
        with open(result_file, "w", newline="") as csvfile:
            fieldnames = [
                "dice_score", "hausdorff_score", "iou_score", 
                "precision_score", "recall_score", "cldice_score",
                "betti_0_score", "betti_1_score", "patient_id"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in self.test_step_outputs:
                result_with_filename = {
                    "dice_score": result["test_dice"],
                    "hausdorff_score": result["test_hausdorff"],
                    "iou_score": result["test_iou"],
                    "precision_score": result["test_precision"],
                    "recall_score": result["test_recall"],
                    "cldice_score": result["test_cldice"],
                    "betti_0_score": result["test_betti_0"],
                    "betti_1_score": result["test_betti_1"],
                    "patient_id": result["patient_id"],
                }
                writer.writerow(result_with_filename)

            # Write summary row
            writer.writerow({
                "dice_score": f"{mean_dice:.4f} ± {np.std(dice_scores):.4f}",
                "hausdorff_score": f"{mean_hausdorff:.4f} ± {np.std(hausdorff_scores):.4f}",
                "iou_score": f"{mean_iou:.4f} ± {np.std(iou_scores):.4f}",
                "precision_score": f"{mean_precision:.4f} ± {np.std(precision_scores):.4f}",
                "recall_score": f"{mean_recall:.4f} ± {np.std(recall_scores):.4f}",
                "cldice_score": f"{mean_cldice:.4f} ± {np.std(cldice_scores):.4f}",
                "betti_0_score": f"{mean_betti_0:.4f} ± {np.std(betti_0_scores):.4f}",
                "betti_1_score": f"{mean_betti_1:.4f} ± {np.std(betti_1_scores):.4f}",
                "patient_id": "AVG ± STD",
            })

        print(f"\nTest Result Summary:")
        print(f"Mean Dice Score: {mean_dice:.4f}")
        print(f"Mean Hausdorff Distance: {mean_hausdorff:.4f}")
        print(f"Mean IoU Score: {mean_iou:.4f}")
        print(f"Mean Precision Score: {mean_precision:.4f}")
        print(f"Mean Recall Score: {mean_recall:.4f}")
        print(f"Mean CLDice Score: {mean_cldice:.4f}")
        print(f"Mean Betti-0 Error: {mean_betti_0:.4f}")
        print(f"Mean Betti-1 Error: {mean_betti_1:.4f}")
        print(f"Detailed result saved to: {result_file}")

        # Clear the outputs
        self.test_step_outputs.clear()


@click.command()
@click.option(
    "--arch_name",
    type=click.Choice(["UNet", "SegResNet", "UNETR", "SwinUNETR"]),
    default="UNETR",
    help="Choose the architecture name for the model.",
)
@click.option(
    "--loss_fn",
    type=click.Choice(["DiceLoss", "DiceCELoss", "DiceFocalLoss"]),
    default="DiceFocalLoss",
    help="Choose the loss function for training.",
)
@click.option(
    "--max_epochs",
    type=int,
    default=300,
    help="Set the maximum number of training epochs.",
)
@click.option(
    "--check_val_every_n_epoch",
    type=int,
    default=10,
    help="Set the frequency of validation checks (in epochs).",
)
@click.option(
    "--gpu_number", type=str, default="0", help="GPU number to use (ex. 0 or 0,1,2,3)"
)
@click.option(
    "--checkpoint_path",
    type=str,
    default=None,
    help="Path to a checkpoint file to load for inference.",
)
@click.option(
    "--guide",
    type=click.Choice(["segMap", "distanceMap"]),
    default="segMap",
    help="Choose the guide for training.",
)
@click.option(
    "--pos",
    type=bool,
    default=False,
    help="Use positional embedding for training.",
)
def main(
    arch_name,
    loss_fn,
    max_epochs,
    check_val_every_n_epoch,
    gpu_number,
    checkpoint_path,
    guide,
    pos
):
    # NCCL communication
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    
    # multiprocessing
    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.set_float32_matmul_precision('medium')
    
    print_monai_config()

    # set up loggers and checkpoints
    log_dir = f"result/proposed_{arch_name}" + ("_distanceMap" if guide == "distanceMap" else "") + ("_pos" if pos else "")
    os.makedirs(log_dir, exist_ok=True)

    # GPU Setting
    if "," in str(gpu_number):
        # Multiple GPUs
        devices = [int(gpu) for gpu in str(gpu_number).split(",")]
        strategy = "ddp_find_unused_parameters_true"
    else:
        # Single GPU
        devices = [int(gpu_number)]
        strategy = "auto" 

    # Set up callbacks
    callbacks = [
        StochasticWeightAveraging(
            swa_lrs=[1e-4], annealing_epochs=5, swa_epoch_start=100
        )
    ]
    dvc_logger = DVCLiveLogger(log_model=True, dir=log_dir, report="html")

    # initialise Lightning's trainer.
    trainer = pytorch_lightning.Trainer(
        devices=devices,
        strategy=strategy,
        max_epochs=max_epochs,
        logger=dvc_logger,  # Use DVC logger instead of CSV logger
        enable_checkpointing=True,
        benchmark=True,
        accumulate_grad_batches=5,
        precision="bf16-mixed",
        check_val_every_n_epoch=check_val_every_n_epoch,
        num_sanity_val_steps=1,
        callbacks=callbacks,
        default_root_dir=log_dir,
    )

    # Initialize data module
    data_module = CoronaryArteryDataModule(
        data_dir="data/imageCAS_heart_pos",
        batch_size=1,
        patch_size=(96, 96, 96),
        num_workers=4,
        cache_rate=0,
        use_distance_map=guide == "distanceMap",
        use_positional_embedding=pos
    )

    if pos:
        print("prepare_data_pos")
        data_module.prepare_data()
    else:
        print("prepare_data")
        data_module.prepare_data()

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint_filename = os.path.basename(checkpoint_path)
        
        if "final_model.ckpt" in checkpoint_filename:
            # start test mode if final epoch checkpoint
            print("Loading final model for testing...")
            model = CoronaryArterySegmentModel.load_from_checkpoint(
                checkpoint_path,
                arch_name=arch_name,
                loss_fn=loss_fn,
                batch_size=1
            )
            model.result_folder = Path(log_dir)  # Set result folder path
            
            # test mode
            trainer.test(model=model, datamodule=data_module)
        
        elif "epoch=" in checkpoint_filename:
            # resume training if intermediate epoch checkpoint
            print("Resuming training from checkpoint...")
            model = CoronaryArterySegmentModel(
                arch_name=arch_name,
                loss_fn=loss_fn,
                batch_size=1
            )
            model.result_folder = Path(log_dir)  # Set result folder path
            
            # resume training (use ckpt_path parameter in trainer.fit)
            trainer.fit(model, datamodule=data_module, ckpt_path=checkpoint_path)
            trainer.save_checkpoint(os.path.join(log_dir, "final_model.ckpt"))
            trainer.test(model=model, datamodule=data_module)
        
        else:
            # test mode if unknown checkpoint format
            print("Unknown checkpoint format, using for testing...")
            model = CoronaryArterySegmentModel.load_from_checkpoint(
                checkpoint_path,
                arch_name=arch_name,
                loss_fn=loss_fn,
                batch_size=1
            )
            model.result_folder = Path(log_dir)  # Set result folder path
            
            trainer.test(model=model, datamodule=data_module)
    else:
        # Initialize model for training
        model = CoronaryArterySegmentModel(arch_name=arch_name, loss_fn=loss_fn, batch_size=1)
        model.result_folder = Path(log_dir)  # Set result folder path

        # Train the model
        trainer.fit(model, datamodule=data_module)
        trainer.save_checkpoint(os.path.join(log_dir, "final_model.ckpt"))
        trainer.test(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()