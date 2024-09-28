import torch
import torch.nn as nn
from diffusers.optimization import get_cosine_schedule_with_warmup
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from dataset import *
from accelerate import Accelerator
import os 
from PIL import Image
from diffusers.utils import make_image_grid
import numpy as np
import argparse
from discriminator import Discriminator, train_discriminator
from torch.utils.tensorboard import SummaryWriter

# Set device (GPU if available, else CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
class VAE(nn.Module):
    def __init__(self, output_channels, img_size, num_channels, num_layers, latent_variable_channels, num_layers_per_block, final_act):
        super(VAE, self).__init__()

        self.output_channels = output_channels
        self.img_size = img_size
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.latent_variable_channels = latent_variable_channels
        self.num_layers_per_block = num_layers_per_block
        dim = int(self.img_size * (1/2)**self.num_layers)
        self.latent_shape = (self.latent_variable_channels, dim, dim)
        encoders_list = []
        batch_norms_enc_list = []
        encoder_res_list = []

        channel_sizes = [output_channels]
        self.fsize = self.img_size


        # List of convolution layers for encoder
        for i in range(self.num_layers):
            if (i == 0):
                channel_sizes.append(self.num_channels)
            else:
                channel_sizes.append(channel_sizes[-1] * 2)
            enc_list_in_block = []
            for j in range(num_layers_per_block):
                
                if j == 0:
                    enc_list_in_block.append(
                        nn.Conv2d(channel_sizes[i], channel_sizes[i+1], 4, 2, 1)
                    )
                else:
                    enc_list_in_block.append(
                        nn.Conv2d(channel_sizes[i+1], channel_sizes[i+1], 4, 1, 'same')
                    )
            encoders_list.append(nn.ModuleList(enc_list_in_block))
            encoder_res_list.append(nn.Conv2d(channel_sizes[i], channel_sizes[i+1], 4, 2, 1))
            batch_norms_enc_list.append(nn.BatchNorm2d(channel_sizes[i+1]))
            
            self.fsize = self.fsize // 2
        self.conv_encoders = nn.ModuleList(encoders_list)
        self.encoder_res = nn.ModuleList(encoder_res_list)
        self.batch_norms_enc = nn.ModuleList(batch_norms_enc_list)
        self.conv_mean = nn.Conv2d(channel_sizes[-1], latent_variable_channels, 4, 1, "same")
        self.conv_std = nn.Conv2d(channel_sizes[-1], latent_variable_channels, 4, 1, "same")
        
        self.first_upsample = nn.ModuleList(
            [
                nn.Conv2d(latent_variable_channels, channel_sizes[-1], 4, 1, 'same'),
                nn.Conv2d(channel_sizes[-1], channel_sizes[-1], 4, 1, 'same'),
            ]

        )
        upsamplers_list = []
        paddings_list = []
        conv_decoders_list = []
        batch_norms_dec_list = []
        decoder_res_list = []

        # List of convolution layers for decoder
        for i in reversed(range(2, len(channel_sizes))):
            upsamplers_list.append(
                nn.UpsamplingNearest2d(scale_factor=2) 
            )
            paddings_list.append(
                nn.ReplicationPad2d(1)
            )
            conv_layers = []
            for j in range(num_layers_per_block):
                if j == 0:
                    conv_layer = nn.Conv2d(channel_sizes[i], channel_sizes[i-1], 3, 1)
                else:
                    conv_layer = nn.Conv2d(channel_sizes[i-1], channel_sizes[i-1], 3, 1, "same")
                conv_layers.append(conv_layer)
            
            decoder_res_list.append(nn.Conv2d(channel_sizes[i], channel_sizes[i-1], 3, 1))
            
            conv_decoders_list.append(
                nn.ModuleList(conv_layers)
            )
            batch_norms_dec_list.append(
                nn.BatchNorm2d(channel_sizes[i-1], 1.e-3)
            )
                
        upsamplers_list.append(
                nn.UpsamplingNearest2d(scale_factor=2) 
            )
        paddings_list.append(
            nn.ReplicationPad2d(1)
        )
        conv_decoders_list.append(
            nn.Conv2d(channel_sizes[1], output_channels, 3, 1)
        )
        self.upsamplers = nn.ModuleList(upsamplers_list)

        self.paddings = nn.ModuleList(paddings_list)
        self.conv_decoders = nn.ModuleList(conv_decoders_list)
        self.batch_norms_dec = nn.ModuleList(batch_norms_dec_list)
        self.res_decoders = nn.ModuleList(decoder_res_list)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.final_act = final_act
        if final_act == "sigmoid":
            self.final_act_fn = nn.Sigmoid()
        elif final_act == "tanh":
            self.final_act_fn = nn.Tanh()
        else:
            self.final_act_fn = lambda x: x/2

    def save(self, path):
        directory = Path(path).parent
        os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), path)
        
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(
                dict(
                    output_channels = self.output_channels,
                    img_size = self.img_size,
                    num_channels = self.num_channels,
                    num_layers = self.num_layers,
                    latent_variable_channels = self.latent_variable_channels,
                    num_layers_per_block = self.num_layers_per_block,
                    final_act=self.final_act
                ),
                f
            )
        print(f"Model saved successfully in {path}")

    @staticmethod
    def load(path):
        directory = Path(path).parent
        config = json.load(open(os.path.join(directory, "config.json")))
        print("Loading config: ", config)
        vae = VAE(**config)
        vae.load_state_dict(torch.load(path))
        return vae
    
    def encode_block(self, x, i):
        res = self.encoder_res[i](x)
        for layer in range(len(self.conv_encoders[i])):
            x = self.conv_encoders[i][layer](x)
            if layer != len(self.conv_encoders[i]) - 1:
                x = self.leakyrelu(x)
        x = self.batch_norms_enc[i](x)
        return self.relu(x + res)

    def encode(self, x):

        for i in range(len(self.conv_encoders)):
            x = self.encode_block(x, i)

       
        return self.conv_mean(x), self.conv_std(x)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def decode_block(self, z, i):
        z = self.upsamplers[i](z)
        z = self.paddings[i](z)
        res = self.res_decoders[i](z)

        for layer in range(len(self.conv_decoders[i])):
            z = self.conv_decoders[i][layer](z)
            if layer != len(self.conv_decoders[i]) - 1:
                z = self.leakyrelu(z)
        z = self.batch_norms_dec[i](z)
        z = z + res
        return z

    def decode(self, z):
        for i in range(len(self.first_upsample)):
            z = self.first_upsample[i](z)
            z = self.leakyrelu(z)
        
        for i in range(len(self.upsamplers) - 1):
            z = self.decode_block(z, i)
        z = self.conv_decoders[-1](self.paddings[-1](self.upsamplers[-1](z)))
        z = self.final_act_fn(z)
        return z

    def get_latent_var(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar

def calculate_adaptive_weight(vgg_loss, gan_loss, last_layer=None):
    if last_layer is not None:
        nll_grads = torch.autograd.grad(vgg_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(gan_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 10).detach()
    return d_weight

    
def print_config():
    print("************** CONFIG *************")
    print(f"{vae.img_size=}\n{vae.latent_variable_channels=}\n{vae.num_layers=}\n{batch_size=}\n{lr=}\n{vae.num_layers_per_block=}\n{vae.num_channels=}\n{vgg_scale=}\n{kl_init_scale=}\n{perceptual_ids=}")
    print("***********************************")

def print_model_weights(model, msg):
    s = 0
    for p in model.parameters():
        s += p.sum().item()
        
    print(f"{msg} -- {s}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', '-i', type=int, default=128, help='Input Image Size')
    parser.add_argument('--batch_size', '-bs', type=int,default=128
                        , help='Batch Size')
    parser.add_argument('--latent_dims', type=int, default=16, help='Width & Height of Latent Space')
    parser.add_argument('--latent_channels', "-lc", type=int, default=4, help='Number of channels in latent space')
    parser.add_argument('--epochs', '-e', type=int, default=20, help='Num Epochs to train')
    parser.add_argument('--load', '-l', type=str, help='Load Filename')
    parser.add_argument('--id', type=str, help='Model ID')

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--num_layers_per_block', '-nlpb', type=int, default=2, help='Num Layers Per Block')
    parser.add_argument('--kl_init_scale', type=float, default=0.05, help='KL Scale')
    parser.add_argument('--num_channels_init', type=int, default=32, help='Num channels for initial conv layer')
    parser.add_argument('--start_epoch', type=int, default=0, help='Num channels for initial conv layer')
    parser.add_argument("--inf", action='store_true')
    parser.add_argument('--mse_scale', type=float, default=0.2, help='Num channels for initial conv layer')
    parser.add_argument('--center_mean_off', action='store_true')
    parser.add_argument('--final_act', type=str, default='tanh')
    parser.add_argument('--adv_scale',type=float, default=0.2)
    parser.add_argument('--load_disc','-ld', type=str)
    parser.add_argument('--loop', type=int, default=1)
    
    

    
    args = parser.parse_args()
    model_id = args.id
    resume_file = args.load
    latent_channels = args.latent_channels
    img_size = args.img_size
    num_layers = int(np.log2(img_size/args.latent_dims))
    num_channel_init = args.num_channels_init
    batch_size = args.batch_size
    epochs = args.epochs
    final_act = args.final_act
    center_mean = not args.center_mean_off

    lr = args.lr
    num_layers_per_block = args.num_layers_per_block
    kl_init_scale = args.kl_init_scale
    mse_scale = args.mse_scale
    anneal_epochs = 2
    perceptual_ids = ["1", "3", "6"] # The modules we will use from the VGG19 model
    start_epoch = args.start_epoch
    adv_scale = args.adv_scale
                    
    accelerator = Accelerator(
        gradient_accumulation_steps=1
    )

    if resume_file:
        vae = VAE.load(resume_file)
        kl_anneal_function = lambda epoch: kl_init_scale
        lr = 1e-4
    else:
        vae = VAE(output_channels=3, img_size=img_size, num_channels=num_channel_init, num_layers=num_layers, latent_variable_channels=latent_channels, num_layers_per_block=num_layers_per_block, final_act=final_act)
    
    train_loader, val_loader = load_data(img_size, batch_size, use_albumenations=True, center_mean=center_mean, test_size=4)
    discriminator = Discriminator()
    disc_loss_fn = nn.MSELoss() # nn.BCELoss()

    if args.load_disc:
        discriminator.load_state_dict(torch.load(args.load_disc))

    if (args.inf):
        vae.eval()
        discriminator.eval()
        vae = vae.to(device)
        discriminator = discriminator.to(device)
        # reconstruct_images(vae, f"reconst_inf", val_loader, f"vae_{model_id}")

        if args.load_disc:
            for data, _ in val_loader:
                with torch.no_grad():
                    data = data.to(device)
                    mu, _ = vae.encode(data)
                    out = vae.decode(mu)
                    disc_real = discriminator(data)
                    disc_fake = discriminator(out)
                    labels_real = torch.full(disc_real.shape, 1, dtype=torch.float, device=device, requires_grad=False)
                    labels_fake = torch.full(disc_fake.shape, 0, dtype=torch.float, device=device, requires_grad=False)    
                    loss_real = disc_loss_fn(disc_real, labels_real)
                    loss_fake = disc_loss_fn(disc_fake, labels_fake)
                    
                    print(f"REAL Vals: {disc_real.mean()}, FAKE Disc: {disc_fake.mean()}")
                    print(f"Loss Real: {loss_real}, Loss fake: {loss_fake}")
                    
        
        exit()
    


    kl_anneal_function = lambda epoch: min(kl_init_scale, kl_init_scale * (epoch + 1) / anneal_epochs)
    vgg = VGGPerceptualLoss(perceptual_ids, center_mean) 


    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr)
    disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=400,
        num_training_steps=(len(train_loader) * epochs)
    )

    vae, optimizer, train_loader, lr_scheduler, vgg, discriminator, disc_optimizer = accelerator.prepare(
        vae, optimizer, train_loader, lr_scheduler, vgg, discriminator, disc_optimizer
    )
    labels_real = None
    labels_fake = None
    num_disc_steps = 0
    num_steps = args.start_epoch * len(train_loader)
    disc_train_interval_fn = lambda x: 1 if x < disc_train_start else 25
    disc_train_start = 1 * len(train_loader)
    dir_name = f"models/vae_{model_id}"
    os.makedirs(dir_name, exist_ok=True)
    writer = SummaryWriter(f'logs/{dir_name}')

    for e in range(start_epoch, epochs):
        kl_scale = kl_anneal_function(e)
        for (batch_idx, (data, _)) in enumerate(train_loader):
            for _ in range(args.loop):
                discriminator.train()
                vae.train()

                num_steps += 1
                with accelerator.accumulate([vae, discriminator]):
                    pred, mean, log_var = vae(data)  
                    
                    details = {}
                    
                    mse_loss = F.mse_loss(pred, data, reduction="mean")
                    vgg_loss = vgg.forward(pred, data)
                    kl_loss = torch.mean(-0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp(), 1))
                    loss = vgg_loss + kl_init_scale * kl_loss + mse_scale * mse_loss
                    details["vgg"] = vgg_loss.item()
                    details["kl"] = kl_loss.item()
                    details["mse"] = mse_loss.item()
                    
                    disc_train_interval = disc_train_interval_fn(num_steps)
                    train_disc = (num_steps % disc_train_interval == 0) and (adv_scale > 0)
                    num_disc_steps = num_disc_steps + int(train_disc)
                    if adv_scale > 0:
                        if num_steps > disc_train_start:  
                            disc_fake = discriminator(pred)
                            labels_real = torch.full(disc_fake.shape, 1, dtype=torch.float, device=device, requires_grad=False)
                            disc_loss = disc_loss_fn(disc_fake, labels_real)
                            adaptive_weight = calculate_adaptive_weight(mse_loss, disc_loss, pred)
                            details["w"] = adaptive_weight
                            details["gen"] = disc_loss.item()
                            loss = loss + adv_scale * adaptive_weight * disc_loss 

                    accelerator.backward(loss)
                    accelerator.clip_grad_norm_(vae.parameters(), 1.0)
                    
                    if train_disc:
                        disc_real, disc_fake, loss_real, loss_fake = \
                            train_discriminator(discriminator, 
                                                real_images=data.detach(), 
                                                fake_images=pred.detach(), 
                                                optimizer=disc_optimizer, 
                                                loss_fn=disc_loss_fn,
                                                accelerator=accelerator)
                        details["loss_real"] = loss_real.item()
                        details["loss_fake"] = loss_fake.item()
                        details["disc_real"] = disc_real
                        details["disc_fake"] = disc_fake
                        
                        
                    optimizer.step()                    
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    details_str = ", ".join([f"{k}: {v if isinstance(v, int) else v:0.3f}" for k, v in details.items()])
                    if batch_idx % 50 == 0:
                        writer.add_scalar('loss', loss.item(), global_step=num_steps)
                        for k, v in details.items():
                            if isinstance(v, float) or isinstance(v, int):
                                writer.add_scalar(k, v, global_step=num_steps)
                        print(f"Epochs={e}, {batch_idx}/{len(train_loader)}, Loss: {loss.item():0.3f}, {details_str}, Disc Steps: {num_disc_steps}")  
                    if batch_idx % 500 == 0:
                        write_grid_of_images(pred.detach()[:16], center_mean=center_mean, id=f"current_{num_steps}", dir=f"vae_{model_id}")
                        model_path = f"{dir_name}/model__{num_steps}.pth"
                        vae.save(model_path)        
                        print("VAE saved in: ", model_path)

                        if adv_scale > 0:
                            disc_dir = f"models/disc_{model_id}"
                            os.makedirs(disc_dir, exist_ok=True)
                            disc_model_path = f"{disc_dir}/disc_{e}.pth"
                            torch.save(discriminator.state_dict(), disc_model_path)
                            print("Discriminator saved in: ", disc_model_path)
                        
                        with torch.no_grad():
                            discriminator.eval()
                            disc_real = discriminator(data[:16])
                            disc_fake = discriminator(pred[:16])
                            print(f"REAL Vals: {disc_real.mean()}, FAKE Disc: {disc_fake.mean()}")
                        

                        
        vae.eval()
        reconstruct_images(vae, f"reconst_{e}", val_loader, f"vae_{model_id}", center_mean)   
        model_path = f"{dir_name}/model__{e}.pth"
        vae.save(model_path)        
        print("VAE saved in: ", model_path)

    writer.close()