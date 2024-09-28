from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.image as mpimg
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers.utils import make_image_grid
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from pathlib import Path
import random

data_dir = "./archive/img_align_celeba" # Your dataset location
label_file = "./archive/text_labels_2.json" # Your dataset location

device = torch.device("mps")

def get_transform(img_size, use_albumenations):
    if use_albumenations:
        # data augmentations
        transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(),
            A.RandomBrightnessContrast(),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(img_size, img_size),
            ToTensorV2
        ])
    return transform

class AugmentedImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, center_mean=True, text_labels=None, return_filename_as_target=False):
        self.center_mean = center_mean
        self.return_filename_as_target = return_filename_as_target
        self.text_labels = json.load(open(label_file)) if text_labels is not None else {}
        super().__init__(root, transform=transform, target_transform=target_transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        file_id = Path(path).name
        if self.text_labels:
            text_labels = self.text_labels.get(file_id, [""])
            text_labels = [t for t in text_labels if len(t.split()) <= 7]
            target = random.choice(text_labels)
        if self.transform is not None:
            # Convert PIL Image to numpy array for Albumentations
            image_np = A.Compose([A.ToFloat(max_value=255)])(image=np.array(image))['image']
            # Apply Albumentations transform
            augmented = self.transform(image=image_np)
            image = augmented['image']

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.center_mean:
            image = image * 2 - 1

        target = path if self.return_filename_as_target else target

        return image, target
    
def load_data(img_size, batch_size, use_albumenations, sample=None, center_mean=True, text_labels=False, train_ratio=1.0, test_size=None, get_filepaths = False):
    
    transform = get_transform(img_size, use_albumenations)
    train_dataset = AugmentedImageFolder(root=data_dir, transform=transform, center_mean=center_mean, text_labels=text_labels, return_filename_as_target=get_filepaths)
    
    # else:
    #     train_dataset = ImageFolder(root=data_dir, transform=transform)

    total_size = len(train_dataset)
    test_size = test_size if test_size is not None else int( (1 - train_ratio) * len(train_dataset))
    train_size = total_size - test_size
    train_set, test_set = random_split(train_dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def load_debug_data(img_size, grayscale=False):
    transform = get_transform(img_size, grayscale)
    test_imgs = [os.path.join(data_dir, "img_align_celeba/000001.jpg"), os.path.join(data_dir, "img_align_celeba/000002.jpg")]
    data = [mpimg.imread(test_img) for test_img in test_imgs]
    data = torch.stack([transform(Image.fromarray(d).convert("RGB")) for d in data])
    return data

def display_image(img):
    img = np.array(img)
    if img.ndim == 3:
        if (img.shape[-1] != 3):
            img = np.transpose(img, [1, 2, 0])
        assert img.shape[-1] in (3, 1)
    plt.imshow(img, cmap="gray")
    plt.axis('off')  # Turn off axis labels
    plt.show()


def save_images(vae, id, latent_size, dir="vae"):
    random_latents = torch.randn(size=[4, *latent_size]).to(device)
    print("LATENT SHAPE", random_latents.shape)
    random_latents = random_latents
    images = []
    with torch.no_grad():
        out = vae.decode(random_latents)
        for img in out:
            img = img.cpu().numpy()
            img = img.transpose([1, 2, 0])
            img = (img + 1)  * 255
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            images.append(img)
    image_grid = make_image_grid(images, rows=2, cols=2)
    test_dir = os.path.join(f"output_dir/{dir}")
    os.makedirs(test_dir, exist_ok=True)
    img_filepath = f"{test_dir}/{id}.png"
    image_grid.save(img_filepath)
    print(f"File saved in {img_filepath}")

def write_grid_of_images(out, center_mean, id, dir="vae"):
    if center_mean:
        out = (out + 1)/2
    out = out.permute([0, 2, 3, 1])
    out = (out * 255).cpu().numpy().astype(np.uint8)
    out = np.clip(out, 0, 255)
    images = [Image.fromarray(img) for img in out]
    rows=len(images)//4
    cols=4
    images = images[:rows*cols]
    image_grid = make_image_grid(images, rows=rows, cols=cols)
    test_dir = os.path.join(f"output_dir/{dir}")
    os.makedirs(test_dir, exist_ok=True)
    img_filepath = f"{test_dir}/{id}.png"
    image_grid.save(img_filepath)
    print(f"File saved in {img_filepath}")

def get_disc_output(vae, discriminator, val_loader,):
    for data, _ in val_loader:
        with torch.no_grad():

            mu, _ = vae.encode(data)
            out = vae.decode(mu)
            disc_real = discriminator(data)
            disc_fake = discriminator(out)
            print(f"REAL Vals: {disc_real.mean()}, FAKE Disc: {disc_fake.mean()}")


def reconstruct_images(vae, id, test_loader, dir="vae", center_mean=True):
    images = []
    for (t, _) in test_loader:
        print(t.max(), t.min())
        with torch.no_grad():
            t = t.to(device)
            mu, _ = vae.encode(t)
            out = vae.decode(mu)
            print(out.max(), out.min())
            if center_mean:
                out = (out + 1)/2
            out = out.permute([0, 2, 3, 1])
            out = (out * 255).cpu().numpy().astype(np.uint8)
            out = np.clip(out, 0, 255)
            images = [Image.fromarray(img) for img in out]
            break
    image_grid = make_image_grid(images, rows=len(images)//4, cols=4)
    test_dir = os.path.join(f"output_dir/{dir}")
    os.makedirs(test_dir, exist_ok=True)
    img_filepath = f"{test_dir}/{id}.png"
    image_grid.save(img_filepath)
    print(f"File saved in {img_filepath}")



# VGG-based Perceptual Loss
class VGGPerceptualLoss:
    def __init__(self, feature_layers = ["1", "3", "6"], center_mean=True):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features
        self.feature_layers = feature_layers
        self.criterion = nn.MSELoss()
        for param in self.vgg19.parameters():
            param.requires_grad = False
        self.center_mean = center_mean

        self.vgg19.to(device).eval()
        

    def forward(self, generated_img, target_img):
        if self.center_mean:
            generated_img = (generated_img + 1)/2
            target_img = (target_img + 1)/2
        gen_features = self.get_features(generated_img)
        target_features = self.get_features(target_img)
        perceptual_loss = 0.0
        for gen_feat, target_feat in zip(gen_features, target_features):
            perceptual_loss += self.criterion(gen_feat, target_feat)
        return perceptual_loss

    def get_features(self, x):
        features = []
        for name, layer in self.vgg19._modules.items():
            x = layer(x)

            if name in self.feature_layers:
                features.append(x)
            if len(features) == len(self.feature_layers):
                break
        return features

def loss_function(vgg, recon_x, x, mean, logvar, vgg_scale, kl_scale, mse_scale=1):
    reconstruction_loss = F.mse_loss(recon_x, x, reduction="mean")
    vgg_loss = vgg.forward(recon_x, x)
    KLD = torch.mean(-0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp(), 1))
    return mse_scale*reconstruction_loss + vgg_scale * vgg_loss + kl_scale * KLD, {
        "mse": reconstruction_loss.item(),
        "vgg": vgg_loss.item(),
        "kl": KLD.item(),
    }

def get_random_labels():
    return ['man',
 'smiling wearing glasses',
 'old woman wearing glasses',
 'woman smiling wearing glasses',
 'young man smiling wearing glasses',
 'man smiling wearing glasses',
 'woman wearing glasses',
 'young man wearing glasses',
 '',
 'man wearing glasses',
 'old woman smiling wearing glasses',
 'young woman smiling wearing glasses',
 'old man',
 'young woman smiling',
 'old man smiling',
 'old man smiling wearing glasses']

if __name__ == "__main__":
    train_loader = load_data(128, 64, True, 64)
    for (t, _) in train_loader:
        t = t.numpy().transpose([0, 2, 3, 1])
        print(t.shape)
        t = (t * 255).astype(np.uint8)
        imgs = [Image.fromarray(x) for x in t]
        image_grid = make_image_grid(imgs, rows=8, cols=8)
        display_image(image_grid)


from PIL import Image, ImageDraw, ImageFont
import os

def create_image_grid_with_annots(images, annotations, rows, cols):
    images = [img.resize((128, 128)) for img in images]
    # Check if the number of images and annotations match
    if len(images) != len(annotations):
        raise ValueError("Number of images and annotations must be the same.")

    # Set the size of each grid cell
    cell_width = max(img.width for img in images)
    cell_height = max(img.height for img in images) + 40  # Additional space for annotations

    # Create a new image for the grid
    grid_width = cell_width * cols
    grid_height = cell_height * rows
    grid_image = Image.new("RGB", (grid_width, grid_height), color="white")

    # Initialize the drawing context
    draw = ImageDraw.Draw(grid_image)

    # Set the font for annotations
    font = ImageFont.load_default()

    # Iterate over images and annotations to fill the grid
    for i, (img, annotation) in enumerate(zip(images, annotations)):
        # Calculate the position of the current cell
        col_idx = i % cols
        row_idx = i // cols
        x = col_idx * cell_width
        y = row_idx * cell_height

        # Paste the image onto the grid
        grid_image.paste(img, (x, y))

        num_spaces = len(annotation.split())
        if num_spaces > 4:
            annotation = " ".join(annotation.split()[:4]) + "\n"+ " ".join(annotation.split()[4:])
        # Write the annotation below the image
        draw.text((x, y + img.height), annotation, fill="black", font=font)

    # Save the grid image
    # grid_image.save(save_path)
    return grid_image