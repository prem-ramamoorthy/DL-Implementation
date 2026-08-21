import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Autoencoder(nn.Module):

    def __init__(self, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 28 * 28), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

transform = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

def train(model, loader, epochs=10, lr=1e-3, denoise=False, noise_std=0.3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for x, _ in loader:                      # labels unused — unsupervised
            x = x.view(x.size(0), -1).to(device)  # flatten 28x28 -> 784

            x_input = x
            if denoise:
                x_input = x + noise_std * torch.randn_like(x)
                x_input = torch.clamp(x_input, 0.0, 1.0)

            x_hat, _ = model(x_input)
            loss = criterion(x_hat, x)             # always reconstruct the CLEAN image

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(loader.dataset)
        print(f"epoch {epoch + 1:2d}/{epochs}  loss={avg_loss:.4f}")

    return model

def plot_reconstructions(model, loader, n=8, denoise=False, noise_std=0.3, filename="reconstructions.png"):
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].view(n, -1).to(device)

    x_input = x
    if denoise:
        x_input = torch.clamp(x + noise_std * torch.randn_like(x), 0.0, 1.0)

    with torch.no_grad():
        x_hat, _ = model(x_input)

    fig, axes = plt.subplots(3 if denoise else 2, n, figsize=(n * 1.2, (3 if denoise else 2) * 1.4))
    for i in range(n):
        row = 0
        if denoise:
            axes[row, i].imshow(x_input[i].cpu().view(28, 28), cmap="gray")
            axes[row, i].set_title("corrupted", fontsize=8)
            axes[row, i].axis("off")
            row += 1
        axes[row, i].imshow(x[i].cpu().view(28, 28), cmap="gray")
        axes[row, i].set_title("original", fontsize=8)
        axes[row, i].axis("off")
        axes[row + 1, i].imshow(x_hat[i].cpu().view(28, 28), cmap="gray")
        axes[row + 1, i].set_title("reconstructed", fontsize=8)
        axes[row + 1, i].axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"saved {filename}")


def plot_latent_space(model, loader, filename="latent_space.png"):
    model.eval()
    zs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.view(x.size(0), -1).to(device)
            _, z = model(x)
            zs.append(z.cpu())
            labels.append(y)
    zs = torch.cat(zs).numpy()
    labels = torch.cat(labels).numpy()

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(zs[:, 0], zs[:, 1], c=labels, cmap="tab10", s=4, alpha=0.6)
    plt.colorbar(scatter, ticks=range(10), label="digit")
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title("Learned 2D latent space (MNIST test set)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"saved {filename}")


if __name__ == "__main__":
    print("=== training plain autoencoder ===")
    ae = Autoencoder(latent_dim=2)
    train(ae, train_loader, epochs=10, denoise=False)
    plot_reconstructions(ae, test_loader, denoise=False, filename="ae_reconstructions.png")
    plot_latent_space(ae, test_loader, filename="ae_latent_space.png")

    print("\n=== training denoising autoencoder ===")
    dae = Autoencoder(latent_dim=2)
    train(dae, train_loader, epochs=10, denoise=True, noise_std=0.3)
    plot_reconstructions(dae, test_loader, denoise=True, noise_std=0.3, filename="dae_reconstructions.png")