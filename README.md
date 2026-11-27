# Text-to-Image Latent Diffusion

A latent diffusion pipeline that generates images from text prompts, with each stage trained separately: an image autoencoder, a text encoder, and a conditional denoising UNet. A Flask API runs generation as background jobs.

The repo is named for the original goal of animated video, but the code currently covers text-to-image generation.

**Stack:** Python, PyTorch, Diffusers, Transformers, Accelerate, Albumentations, Flask

## How it works

```mermaid
flowchart LR
  P[Text prompt] --> T[DistilBERT text encoder]
  T --> U[Conditional UNet with DDPM scheduler]
  U --> D[VAE decoder]
  D --> I[Generated image]
```

1. **VAE:** compresses images into a small latent space, trained with a KL term, VGG perceptual loss and an adversarial loss from a discriminator.
2. **Text encoder:** DistilBERT trained with a CLIP-style contrastive loss to align text with image latents.
3. **Conditional UNet:** learns to denoise latents given the text embedding, using Diffusers' `UNet2DConditionModel` and `DDPMScheduler`.
4. **API:** runs sampling in a separate process and saves intermediate images as it goes, so a client can poll a job and watch the image form.

## Files

| File | Purpose |
|---|---|
| `vae.py` | VAE model and training |
| `discriminator.py` | Discriminator for the adversarial loss |
| `train_clip.py` | Text encoder and contrastive training |
| `conditional_ldm.py` | `TextToImage` model, UNet training and inference |
| `dataset.py` | Data loading, augmentation and perceptual loss |
| `api.py` | Flask job API |

## API

| Endpoint | Purpose |
|---|---|
| `POST /text_to_image` | Starts a job from `prompt`, with optional `n_images`, `n_inf_steps` and `stream_freq`; returns a `job_id` |
| `GET /jobs/<job_id>/` | Returns the latest images and whether the job has finished |

## Setup

```bash
pip install -r requirements.txt
```

Each script picks the device itself: CUDA if available, then Apple Silicon (`mps`), then CPU.

## Current limitations

The module-name and syntax errors are fixed, but you still need to supply the data and weights:

- **Not included:** model checkpoints and the dataset. `dataset.py` expects CelebA images under `./archive/img_align_celeba` and text labels in `./archive/text_labels_2.json`.
- **Unpinned dependencies:** `requirements.txt` lists the packages the code imports, without versions.
