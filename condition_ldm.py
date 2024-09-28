import torch
import torch.nn as nn
from diffusers import DDPMScheduler, UNet2DConditionModel, DDIMScheduler
import numpy as np
import os
from PIL import Image
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch.nn.functional as F
from diffusers.utils import make_image_grid
import os
import argparse
import time
from vae import VAE
from train_clip import TextEncoder
from dataset import *
import tqdm
from torch.utils.tensorboard import SummaryWriter


# Some default hyperparameter setups
NUM_UNET_LAYERS = 3
NOISE_SCHEDULER_TYPE = "ddim"
USE_ALBUMENTATIONS = True

if NUM_UNET_LAYERS == 2:
    DOWN_BLOCK_TYPES = ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D")
    UP_BLOCK_TYPES = ("CrossAttnUpBlock2D", "CrossAttnUpBlock2D")
    BLOCK_OUT_CHANNELS = (48, 96)
elif NUM_UNET_LAYERS == 3:
    DOWN_BLOCK_TYPES = ("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D")
    UP_BLOCK_TYPES = ("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D")
    BLOCK_OUT_CHANNELS = (64, 128, 256)
elif NUM_UNET_LAYERS == 4:
    DOWN_BLOCK_TYPES = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D")
    UP_BLOCK_TYPES = ("CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
    BLOCK_OUT_CHANNELS = (64, 128, 256, 256) # (128, 256, 512, 768)

class TextToImage(nn.Module):
    def __init__(self, vae, clipmodel, unet_model=None, noise_scheduler=None, scale_factor=1):
        super(TextToImage, self).__init__()

        self.vae = vae
        self.latent_shape = self.vae.latent_shape
        self.text_model =  clipmodel
        self.scale_factor = scale_factor
        if unet_model is None:
            self.unet_model = UNet2DConditionModel(sample_size=self.latent_shape[1], # Dimensions of latent space
                                in_channels=self.latent_shape[0], # Num channels in latent space
                                out_channels=self.latent_shape[0], # Num channels in latent space
                                down_block_types=DOWN_BLOCK_TYPES,
                                up_block_types=UP_BLOCK_TYPES,
                                block_out_channels=BLOCK_OUT_CHANNELS,
                                norm_num_groups=16,
                                layers_per_block=args.nlpb
                                )
        else:   
            self.unet_model = unet_model
        if noise_scheduler is None:
            if NOISE_SCHEDULER_TYPE == "ddpm":
                self.noise_scheduler = DDPMScheduler(num_train_timesteps=num_steps, 
                                                beta_schedule='squaredcos_cap_v2'
                                                )
            else:
                self.noise_scheduler = DDIMScheduler(
                    num_train_timesteps=num_steps,
                    beta_schedule="squaredcos_cap_v2"
                )
        else:
            self.noise_scheduler = noise_scheduler
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        self.text_model.eval()
        for param in self.text_model.parameters():
            param.requires_grad = False
    
    def save(self, path, unet_path="0"):
        # Save VAE
        os.makedirs(path, exist_ok=True)

        vae_path = Path(path)/"vae/vae.pth"
        if not os.path.exists(vae_path):
            self.vae.save(vae_path)

        # Save CLIP
        clip_path = Path(path)/"clip/clip.pth"
        if not os.path.exists(clip_path):
            self.text_model.save(clip_path)

        # Save UNET
        unet_path = Path(path)/"unet"/unet_path
        self.unet_model.save_pretrained(unet_path)
        # Save scheduler

        schedler_path = Path(path)/"noise_scheduler"
        self.noise_scheduler.save_pretrained(schedler_path)

        json.dump({"scale_factor": self.scale_factor, "noise_scheduler_type": NOISE_SCHEDULER_TYPE}, open(Path(path)/"additional_config.json", "w"))

        print(f"Saved! {vae_path=}, {clip_path=}, {schedler_path=}, {unet_path=}")

    @staticmethod
    def create(vae_path, clip_path, scale_factor):
        vae = VAE.load(vae_path)
        clip = TextEncoder.load(clip_path)
        return TextToImage(vae, clip, None, None, scale_factor)

    @staticmethod
    def load(path):
        global NOISE_SCHEDULER_TYPE
        parent_path = Path(path).parent.parent
        # Load VAE
        vae = VAE.load(Path(parent_path)/"vae/vae.pth")

        # Load CLIP
        text_model = TextEncoder.load(Path(parent_path)/"clip/clip.pth")

        # Load UNET
        unet_model = UNet2DConditionModel.from_pretrained(path)

        # Load Scheduler
        noise_scheduler = DDPMScheduler.from_pretrained(Path(parent_path)/"noise_scheduler")
        additional_config = json.load(open(Path(parent_path)/"additional_config.json"))
        NOISE_SCHEDULER_TYPE = additional_config.get("noise_scheduler_type", "ddpm")
        return TextToImage(vae, text_model, unet_model, noise_scheduler, additional_config["scale_factor"])
        
        
    def forward(self, noisy_image, text_data, timesteps):

        # Run CLIP Encoder on Text
        text_encoding, attn_mask = self.text_model(text_data)

        # Run UNET to generate noise for the current timestep
        noise_pred = self.unet_model(noisy_image, timesteps, 
                                        encoder_hidden_states=text_encoding,
                                        encoder_attention_mask=attn_mask,
                                        return_dict=False)[0]
        return noise_pred

    def train_step(self, imgs, text_data):

        # Run VAE Encoder - Note that we don't run the reparameterization stuff. Just the "mean" outputted by the network.
        data, _ = self.vae.encode(imgs)
        # Scale the latents before diffusion training. This is optional but it is good for stable training.
        data = data * self.scale_factor
        
        data_shape = data.shape

        # Generate random noise
        noise = torch.randn(data_shape, device=device)

        # Generate random timesteps
        timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (data_shape[0],), device=device,
                dtype=torch.int64
            )
        
        # Forward Diffusion
        noisy_image = self.noise_scheduler.add_noise(data, noise, timesteps)

        # Call U-Net
        noise_pred = self(noisy_image, text_data, timesteps)
        return noise_pred, noise
    
    @torch.no_grad()
    def run_pipeline(self, prompts, num_inference_steps=None, center_mean=True, stream_freq=None, image=None, start_timestep_factor=0):
        def post(img):
            img = img.cpu().numpy().transpose([0, 2, 3, 1])
            if center_mean:
                img = (img + 1)/2
            img = (img * 255).astype(np.uint8)
            img = [Image.fromarray(i) for i in img]
            return img
        
        def run_vae(img):
            img = img / self.scale_factor
            img = self.vae.decode(img)
            return post(img)

            

        if num_inference_steps is None or num_inference_steps > self.noise_scheduler.config.num_train_timesteps:
            num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        
        batch_size = len(prompts)

        generator = torch.Generator(device="cpu").manual_seed(np.random.randint(1, 100))    
        if image is None:
            image = torch.randn(size=(batch_size, *self.latent_shape), generator=generator).to(device)
        else:
            if (image.ndim != 4):
                image = np.array([image for _ in range(batch_size)])
            image = torch.tensor(image).to(device).float()

        self.noise_scheduler.set_timesteps(num_inference_steps)
        self.unet_model.eval()
        factor = int((start_timestep_factor) * self.noise_scheduler.timesteps.shape[0])
        timesteps = self.noise_scheduler.timesteps[factor:]
        idx = 0
        for idx, t in enumerate(tqdm.tqdm(timesteps)):
            # Call U-Net to generate noise for the current timesteps
            model_output = self(image, prompts, t)

            # Reverse Diffusion
            image = self.noise_scheduler.step(model_output, t, image, generator=generator).prev_sample
            if stream_freq is not None:
                if t % stream_freq == 0:
                    yield run_vae(image), {"t": idx, "total_t": num_inference_steps, "latent": image.cpu().numpy()}
        yield run_vae(image), {"t": idx, "total_t": num_inference_steps, "latent": image.cpu().numpy()}

def evaluate(id):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    random_labels = get_random_labels()

    images = next(diffusion_model.run_pipeline(
        random_labels,
        stream_freq=None
    ))[0]

    image_grid = create_image_grid_with_annots(images, random_labels, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(f"output_dir/t2i_{model_id}")
    os.makedirs(test_dir, exist_ok=True)
    img_filename = f"{test_dir}/{id}.png"
    image_grid.save(img_filename)
    print("SAVED in "+ img_filename)


device = torch.device("mps")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', "-ns", type=int, default=400, help='Description of argument 3')
    parser.add_argument('--vae', type=str, help='VAE file')
    parser.add_argument('--clipmodel', '-cm', type=str, help='Text Model')
    parser.add_argument('--batch_size', '-bs', type=int,default=128
                        , help='Batch Size')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Num Epochs to train')
    parser.add_argument('--load', '-l', type=str, help='Load Filename')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--model_id', type=str, help='Learning Rate')
    parser.add_argument('--start_epoch', type=int, default=0, help='Num channels for initial conv layer')
    parser.add_argument('--center_mean_off', action='store_true')
    parser.add_argument('--inf', action='store_true', help='Run Inference')
    parser.add_argument('-nlpb', type=int, default=2)
    

    args = parser.parse_args()

    model_id = args.model_id
    
    load_model_file = args.load
    batch_size = args.batch_size
    num_steps = args.num_steps
    epochs = args.epochs
    lr = args.lr
    center_mean = not args.center_mean_off
    scale_factor = 0.5

    dirname = f"models/cldm_{model_id}"
    os.makedirs(dirname, exist_ok=True)

    if args.load:
        diffusion_model = TextToImage.load(args.load)
    else:
        diffusion_model = TextToImage.create(args.vae, args.clipmodel, scale_factor=scale_factor)
        
        diffusion_model.save(dirname)

    img_size = diffusion_model.vae.img_size
    train_loader, _ = load_data(img_size, batch_size, USE_ALBUMENTATIONS, center_mean=center_mean, text_labels=True)
    nlpb = args.nlpb

    optimizer = torch.optim.AdamW(diffusion_model.unet_model.parameters(), lr=lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_loader) * epochs)
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=1
    )
    diffusion_model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        diffusion_model, optimizer, train_loader, lr_scheduler
    )
    
    if args.inf:
        diffusion_model.eval()
        evaluate("test")
        exit()
    writer = SummaryWriter(f'logs/diffusion_{dirname}')

    n_steps = args.start_epoch * len(train_loader)
    for iteration in range(args.start_epoch, epochs):
        avg_loss = []
        for batch_idx, (imgs, text_data) in tqdm.tqdm(enumerate(train_loader)):
            diffusion_model.unet_model.train()
            n_steps += 1
            for _ in range(1):
                with accelerator.accumulate(diffusion_model):
                    noise_pred, noise = diffusion_model.train_step(imgs, text_data)

                    # Mean Square error between generated noise and predicted noise
                    loss = F.mse_loss(noise_pred, noise)

                    # Gradients and Backprop
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    avg_loss.append(loss.item())
                    if n_steps % 1000 == 0:
                        print(f"{iteration}, {batch_idx}/{len(train_loader)} , {np.mean(avg_loss):.4f}")
                        writer.add_scalar('loss', np.mean(avg_loss), global_step=n_steps)
                        avg_loss = []

                    if n_steps % 1000 == 0:
                        evaluate(f"s{n_steps:06d}")
    
        diffusion_model.save(dirname, f"{n_steps}")
        evaluate(f"{iteration:04d}")

    writer.close()
