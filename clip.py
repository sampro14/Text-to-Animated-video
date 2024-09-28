import os
from accelerate import Accelerator

import argparse
import time
from vae import VAE
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np
from dataset import load_data
import json
from pathlib import Path

'''
The CLIP model is generally trained with a Vision Transformer as the image encoder, but in this project I just used an already trained
VAE model (see vae.py).
'''

os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = torch.device("mps")

class TextEncoder(nn.Module):
    def __init__(self, img_size=256, out_size=1280):
        super(TextEncoder, self).__init__()
        text_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        for name, params in text_model.named_parameters():
            # Freeze all but the last layer of the pretrained distilbert model
            if "transformer.layer.5" in name:
                params.requires_grad = True
            else:
                params.requires_grad = False

        self.text_model = text_model
        self.img_size = img_size
        self.out_size = out_size
        self.text_linear_layer = nn.Linear(768, out_size)
        self.img_linear_layer = nn.Linear(img_size, out_size)
        self.t = torch.nn.Parameter(torch.tensor(0.5, requires_grad=True))

    def save(self, path):
        directory = Path(path).parent

        os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), path)
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(
                dict(
                    img_size = int(self.img_size),
                    out_size = int(self.out_size)
                ),
                f
            )
        print(f"Model saved successfully in {path}")
    
    @staticmethod
    def load(path):
        directory = Path(path).parent
        config = json.load(open(os.path.join(directory, "config.json")))
        print("Loading config: ", config)
        model = TextEncoder(**config)
        model.load_state_dict(torch.load(path))
        return model

    def clamp_t(self):
        self.t.data = torch.clamp(self.t.data, max=10)

    def image_forward(self, image):
        image = image.view(image.shape[0], -1)
        return self.img_linear_layer(image)

    def forward(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
        attention_mask = (input_ids != 0)
        text_embeddings = self.text_model(input_ids, attention_mask).last_hidden_state
        text_embeddings = self.text_linear_layer(text_embeddings)
        return text_embeddings, attention_mask

def clip_loss(I_f, T_f, t, print_matrix=False):
    '''
    See CLIP paper for information on loss:
    https://arxiv.org/abs/2103.00020
    '''
    I_e = F.normalize(I_f, p=2, dim=-1)
    T_e = F.normalize(T_f, p=2, dim=-1)
    logits = torch.matmul(T_e, I_e.t()) * torch.exp(t)
    
    labels = torch.arange(I_e.shape[0]).to(device)
    loss_i = F.cross_entropy(logits.T, labels, reduction='none')
    loss_t = F.cross_entropy(logits, labels, reduction='none')

    if print_matrix:
        print(f"{logits.cpu()}")
    loss = (1 * loss_i + 1 * loss_t).mean()/2

    return loss

@torch.no_grad
def evaluate(test_dataloader, print_matrix=False):
    total_contrastive_loss = 0
    total_size = 0
    for idx, (img, text) in enumerate(test_dataloader):
        encodings, _ = model(text)
        img = vae.reparametrize(*vae.encode(img))

        image_enc = model.image_forward(img)
        encodings_mean = torch.mean(encodings, axis=1)
        contrastive_loss = clip_loss(image_enc, encodings_mean, model.t, True if idx == 0 else print_matrix)
        total_contrastive_loss += (contrastive_loss.cpu().numpy()) * img.shape[0]
        total_size += img.shape[0]
    return total_contrastive_loss/total_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--vae', type=str, help='VAE file')
    parser.add_argument('--batch_size', '-bs', type=int,default=128
                        , help='Batch Size')
    parser.add_argument('--epochs', '-e', type=int, default=20, help='Num Epochs to train')
    parser.add_argument('--load', '-l', type=str, help='Load Filename')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--model_id', default = "", type=str, help='Learning Rate')
    parser.add_argument('--start_epoch', type=int, default=0, help='Num channels for initial conv layer')
    parser.add_argument('--center_mean_off', action='store_true')
    parser.add_argument('--inf', action='store_true', help='Run Inference')
    parser.add_argument('--dim', type=int, default=1280, help='Dim')

    
    args = parser.parse_args()
    vae = VAE.load(args.vae)
    latent_shape = vae.latent_shape
    batch_size = args.batch_size
    scaling_factor = 1

    model = TextEncoder(np.prod(latent_shape), args.dim)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader, test_loader = load_data(128, batch_size, True, None, True, text_labels=True, test_size=256)
    accelerator = Accelerator(
        gradient_accumulation_steps=1
    )
    vae, model, optimizer, train_loader, test_loader = accelerator.prepare(
        vae, model, optimizer, train_loader, test_loader
    )
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    
    model_path = f"models/clip_model_{args.model_id}"
    os.makedirs(model_path, exist_ok=True)

    if args.load:
        model.load_state_dict(torch.load(args.load))
        print("LOADED WEIGHTS")
        print(evaluate(test_loader, print_matrix=True))

    for epoch in range(args.epochs):  # Number of epochs
        for iter, (img_data, text_data) in enumerate(train_loader):

            optimizer.zero_grad()
            
            # Text encoder
            encodings, _ = model(text_data)

            # image embeddings
            imgs = vae.encode(img_data)[0] * scaling_factor
            image_enc = model.image_forward(imgs)

            encodings_mean = torch.mean(encodings, axis=1)
            contrastive_loss = clip_loss(image_enc, encodings_mean, model.t)
            accelerator.backward(contrastive_loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)            
            optimizer.step()
            model.clamp_t()
            if iter % 50 == 0:
                print(f"Epoch {epoch + 1}, Iter: {iter + 1}/{len(train_loader)}, Loss: {contrastive_loss.item():.3f}, {model.t.detach().item():.3f}")
        eval_loss = evaluate(test_loader)
        print(f"Epoch: {epoch + 1}, Loss: {eval_loss}")
        model.save(f"{model_path}/model_{epoch}.pth")
        print(f"Saved model in {model_path}/model_{epoch}.pth")