import torch
import torch.nn as nn

device = torch.device("mps")

bce_loss = nn.BCELoss()
class Discriminator(nn.Module):
    def __init__(self, image_channels=3, num_filters_last=64, n_layers=3):
        super(Discriminator, self).__init__()

        layers = [nn.Conv2d(image_channels, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                          2 if i < n_layers else 1, 1, bias=False),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return nn.Sigmoid()(self.model(x))
    
def train_discriminator(discriminator, real_images, fake_images, optimizer, loss_fn, accelerator):
    disc_real = discriminator(real_images.detach())
    disc_fake = discriminator(fake_images.detach())
    labels_real = torch.full(disc_real.shape, 1, dtype=torch.float, device=device)
    labels_fake = torch.full(disc_fake.shape, 0, dtype=torch.float, device=device)    
    loss_real = loss_fn(disc_real, labels_real)
    loss_fake = loss_fn(disc_fake, labels_fake)
    disc_loss = (loss_real + loss_fake)/2
    accelerator.backward(disc_loss)
    accelerator.clip_grad_norm_(discriminator.parameters(), 1.0)

    optimizer.step()
    optimizer.zero_grad()
    return disc_real.mean(), disc_fake.mean(), loss_real, loss_fake