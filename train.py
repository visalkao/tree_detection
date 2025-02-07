# train.py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from monai.transforms import Compose, EnsureChannelFirstd, ScaleIntensityd, ToTensor, Activations, AsDiscrete

import sys
if sys.platform == 'win32':
    torch.multiprocessing.set_start_method('spawn', force=True)
    
import logging
import tempfile
from glob import glob
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import monai
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
# from monai.transforms import (
#     Activations,
#     AsDiscrete,
#     Compose,
#     LoadImage,
#     RandRotate90,
#     RandSpatialCrop,
#     ScaleIntensity,
# )
from monai.visualize import plot_2d_or_3d_image
from monai.losses import DiceLoss
# from monai.metrics import DiceMetric
# from monai.transforms import (Compose, LoadImage, EnsureChannelFirstd, ScaleIntensityd, ToTensor, 
#                             RandFlip, RandRotate, Resize)


class VerifiedDataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset
        self._verify_dataset()
        
    def _verify_dataset(self):
        """Validate all entries before training"""
        print("Verifying dataset integrity...")
        for i in range(len(self.ds)):
            try:
                img = self.ds[i]['image']
                lbl = self.ds[i]['label']
                if not isinstance(img, np.ndarray):
                    img = np.array(img)
                if not isinstance(lbl, np.ndarray):
                    lbl = np.array(lbl)
                assert isinstance(img, np.ndarray), "Image not numpy array"
                assert isinstance(lbl, np.ndarray), "Label not numpy array"
                assert img.shape == (256, 256), f"Invalid image shape: {img.shape}"
                assert lbl.shape == (256, 256), f"Invalid label shape: {lbl.shape}"
            except Exception as e:
                raise RuntimeError(f"Invalid data at index {i}: {str(e)}") from e
        print("Dataset verification complete.")

    def __getitem__(self, index):
        img = self.ds[index]['image']
        lbl = self.ds[index]['label']
        
        # Convert to float32 and scale
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        if not isinstance(lbl, np.ndarray):
            lbl = np.array(lbl)
        data = {
            'image': img.astype(np.float32),
            'label': lbl.astype(np.float32)
        }
        
        transforms = Compose([
            EnsureChannelFirstd(keys=['image', 'label'], channel_dim="no_channel"),
            ScaleIntensityd(keys=['image']),
            ToTensor()
        ])
        
        return transforms(data)['image'], transforms(data)['label']

    def __len__(self):
        return len(self.ds)

def main():
    # Windows-specific multiprocessing setup
    torch.multiprocessing.freeze_support()
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Load verified dataset
    ds = load_from_disk("./processed_tcd_final")
    train_ds = VerifiedDataset(ds['train'])
    val_ds = VerifiedDataset(ds['test'])
    
    # Conservative DataLoader config
    loader_config = {
        'batch_size': 8,
        'num_workers': 0 if sys.platform == 'win32' else 2,
        'persistent_workers': False,
        'pin_memory': True,
        'prefetch_factor': None
    }
    
    train_loader = DataLoader(train_ds, shuffle=True, **loader_config)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_config)
    
    # Your existing model and training loop
    # ...
    

    # Create DataLoaders
    # train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    # val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
    torch.multiprocessing.freeze_support()
    torch.multiprocessing.set_start_method('spawn', force=True)
    # Metrics and post-processing (unchanged)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64),
        strides=(2, 2),
    ).to(device)
    loss_function = monai.losses.DiceLoss(sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()



    # train_loader, val_loader = create_train_val_dataloaders(images_dir=images_dir, labels_dir=labels_dir)

    print("Trainloader:")
    print(train_loader)

    for epoch in range(10):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{10}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            print(f"epoch {epoch + 1}/{10} batch = ")
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    roi_size = (96, 96)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_segmentation2d_array.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="output")
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()
    # torch.save(model.state_dict(), "trained_model.pth")

if __name__ == '__main__':
    main()