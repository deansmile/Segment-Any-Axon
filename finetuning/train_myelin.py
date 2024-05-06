from pathlib import Path
import numpy as np

import torch

import torch_em
from torch_em.model import UNETR
from torch_em.loss import DiceBasedDistanceLoss
from torch_em.transform.label import PerObjectDistanceTransform

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model

data_path = "/scratch/ds5725/micro-sam/data"

def get_dataloader(split, patch_shape, batch_size, train_instance_segmentation):
    assert split in ("train", "val")

    train_image_dir="/scratch/ds5725/micro-sam/data/train/image_tif"
    train_segmentation_dir="/scratch/ds5725/micro-sam/data/train/myelin_mask_tif"
    val_image_dir="/scratch/ds5725/micro-sam/data/val/image_tif"
    val_segmentation_dir="/scratch/ds5725/micro-sam/data/val/myelin_mask_tif"
    # Set directories based on the split
    if split == "train":
        image_dir = train_image_dir
        segmentation_dir = train_segmentation_dir
    else:
        image_dir = val_image_dir
        segmentation_dir = val_segmentation_dir

    raw_key, label_key = "*.tif", "*.tif"

    if train_instance_segmentation:
        # Computes the distance transform for objects to perform end-to-end automatic instance segmentation.
        label_transform = PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False,
            foreground=True, instances=True, min_size=25
        )
    else:
        label_transform = torch_em.transform.label.connected_components

    loader = torch_em.default_segmentation_loader(
        raw_paths=image_dir, raw_key=raw_key,
        label_paths=segmentation_dir, label_key=label_key,
        patch_shape=patch_shape, batch_size=batch_size,
        ndim=2, is_seg_dataset=True,
        label_transform=label_transform,
        num_workers=8, shuffle=True, raw_transform=sam_training.identity,
    )
    return loader


def export_model(checkpoint_name, model_type):
    """Export the trained model."""
    # export the model after training so that it can be used by the rest of the micro_sam library
    export_folder = Path().absolute()
    export_path = export_folder.joinpath("finetuned_hela_model_myelin.pth")
    checkpoint_path = export_folder.joinpath("checkpoints", checkpoint_name, "best.pt")
    export_custom_sam_model(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        save_path=export_path,
    )

def run_training(checkpoint_name, model_type, train_instance_segmentation=True):
    """Run the actual model training."""

    # All hyperparameters for training.
    batch_size = 1  # the training batch size
    patch_shape = (1, 256, 256)  # the size of patches for training
    n_objects_per_batch = 10  # the number of objects per batch that will be sampled
    device = torch.device("cuda")  # the device/GPU used for training
    n_iterations = 1000  # how long we train (in iterations)

    # Get the dataloaders.
    train_loader = get_dataloader("train", patch_shape, batch_size, train_instance_segmentation)
    val_loader = get_dataloader("val", patch_shape, batch_size, train_instance_segmentation)

    # Get the segment anything model
    model = sam_training.get_trainable_sam_model(model_type=model_type, device=device)

    # This class creates all the training data for a batch (inputs, prompts and labels).
    convert_inputs = sam_training.ConvertToSamInputs(transform=model.transform, box_distortion_factor=0.025)

    # Get the optimizer and the LR scheduler
    if train_instance_segmentation:
        # for instance segmentation, we use the UNETR model configuration.
        unetr = UNETR(
            backbone="sam", encoder=model.sam.image_encoder, out_channels=3, use_sam_stats=True,
            final_activation="Sigmoid", use_skip_connection=False, resize_input=True,
        )
        # let's get the parameters for SAM and the decoder from UNETR
        joint_model_params = [params for params in model.parameters()]  # sam parameters
        for name, params in unetr.named_parameters():  # unetr's decoder parameters
            if not name.startswith("encoder"):
                joint_model_params.append(params)
        unetr.to(device)
        optimizer = torch.optim.Adam(joint_model_params, lr=1e-5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=10, verbose=True)

    # the trainer which performs training and validation (implemented using "torch_em")
    if train_instance_segmentation:
        instance_seg_loss = DiceBasedDistanceLoss(mask_distances_in_bg=True)
        trainer = sam_training.JointSamTrainer(
            name=checkpoint_name, train_loader=train_loader, val_loader=val_loader, model=model,
            optimizer=optimizer, device=device, lr_scheduler=scheduler, logger=sam_training.JointSamLogger,
            log_image_interval=100, mixed_precision=True, convert_inputs=convert_inputs,
            n_objects_per_batch=n_objects_per_batch, n_sub_iteration=8, compile_model=False, unetr=unetr,
            instance_loss=instance_seg_loss, instance_metric=instance_seg_loss
        )
    else:
        trainer = sam_training.SamTrainer(
            name=checkpoint_name, train_loader=train_loader, val_loader=val_loader, model=model,
            optimizer=optimizer, device=device, lr_scheduler=scheduler, logger=sam_training.SamLogger,
            log_image_interval=100, mixed_precision=True, convert_inputs=convert_inputs,
            n_objects_per_batch=n_objects_per_batch, n_sub_iteration=8, compile_model=False
        )
    trainer.fit(n_iterations)

model_type = "vit_b_lm"

# The name of the checkpoint. The checkpoints will be stored in './checkpoints/<checkpoint_name>'
checkpoint_name = "sam_hela_myelin"

# Train an additional convolutional decoder for end-to-end automatic instance segmentation
train_instance_segmentation = True

run_training(checkpoint_name, model_type, train_instance_segmentation)
export_model(checkpoint_name, model_type)