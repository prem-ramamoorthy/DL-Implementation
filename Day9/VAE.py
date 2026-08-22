import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 1e-3
LATENT_DIM = 20

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class VAE(nn.Module):

    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)

        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        z = mu + eps * std

        return z


    def forward(self, x):

        x = x.view(x.size(0), -1)

        h = self.encoder(x)

        mu = self.fc_mu(h)

        logvar = self.fc_logvar(h)

        z = self.reparameterize(mu, logvar)

        reconstruction = self.decoder(z)

        return reconstruction, mu, logvar

def vae_loss(reconstruction, x, mu, logvar):

    x = x.view(x.size(0), -1)

    reconstruction_loss = F.binary_cross_entropy(
        reconstruction,
        x,
        reduction="sum"
    )

    kl_loss = -0.5 * torch.sum(
        1
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    total_loss = reconstruction_loss + kl_loss

    return total_loss

model = VAE(
    input_dim=784,
    hidden_dim=400,
    latent_dim=LATENT_DIM
).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


model.train()

for epoch in range(EPOCHS):

    total_loss = 0

    for images, _ in train_loader:

        images = images.to(DEVICE)

        reconstruction, mu, logvar = model(images)

        loss = vae_loss(
            reconstruction,
            images,
            mu,
            logvar
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader.dataset)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f}"
    )

model.eval()

with torch.no_grad():

    images, _ = next(iter(train_loader))

    images = images.to(DEVICE)

    reconstruction, _, _ = model(images)

    reconstruction = reconstruction.view(
        -1, 1, 28, 28
    )

fig, axes = plt.subplots(2, 8, figsize=(12, 3))

for i in range(8):

    axes[0, i].imshow(
        images[i].cpu().view(28, 28),
        cmap="gray"
    )

    axes[0, i].axis("off")

    axes[1, i].imshow(
        reconstruction[i].cpu().view(28, 28),
        cmap="gray"
    )

    axes[1, i].axis("off")

axes[0, 0].set_title("Original")
axes[1, 0].set_title("Reconstructed")

plt.tight_layout()
plt.show()


with torch.no_grad():

    z = torch.randn(
        16,
        LATENT_DIM
    ).to(DEVICE)

    generated = model.decoder(z)

    generated = generated.view(
        -1, 1, 28, 28
    )

fig, axes = plt.subplots(2, 8, figsize=(12, 3))

for i in range(16):

    row = i // 8
    col = i % 8

    axes[row, col].imshow(
        generated[i].cpu().view(28, 28),
        cmap="gray"
    )

    axes[row, col].axis("off")

plt.tight_layout()
plt.show()