import os
import torch
from torch import nn, optim
import torchsde
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


def true_density(x: np.ndarray) -> np.ndarray:
    const1 = 0.5 * 1 / np.sqrt(np.pi)
    return const1 * np.exp(-(x - 1.0) ** 2) + const1 * np.exp(-(x + 1.0) ** 2)


class SDE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.LayerNorm(16),
            nn.Linear(16, 1)
        )

    def f(self, t, y):
        t_expanded = t.expand(y.size(0), 1)
        return self.net(torch.cat([t_expanded, y], dim=1))

    def g(self, t, y):
        return self.sigma * torch.ones_like(y)


class MmdPenalty(nn.Module):
    def forward(self, x1, x1_copy):
        device = x1.device
        c0 = torch.tensor(2.0, device=device)
        c1 = torch.tensor(0.5, device=device)
        c2 = torch.tensor(1.0, device=device)
        term1 = torch.exp(-(x1 - x1_copy) ** 2)
        term2 = -(1 / torch.sqrt(c0)) * torch.exp(-c1 * (c2 + x1 ** 2)) * torch.cosh(x1)
        term3 = -(1 / torch.sqrt(c0)) * torch.exp(-c1 * (c2 + x1_copy ** 2)) * torch.cosh(x1_copy)
        return torch.mean(term1 + term2 + term3)


def plot(sample0, sample1, sample2, sample3):
    os.makedirs("img", exist_ok=True)
    x_plot = np.linspace(-4.5, 4.5, 150)
    density = true_density(x_plot)

    plt.figure(figsize=(16, 3))
    for i, samples in enumerate([sample0, sample1, sample2, sample3]):
        plt.subplot(1, 4, i + 1)
        plt.hist(samples, bins=100, range=(-4.5, 4.5), density=True)
        if i == 3:
            plt.plot(x_plot, density)

    plt.savefig("img/bimodal_1d_gauss.png")


def main():
    epoch = 5000
    batch_size = 128
    state_size = 1
    t_size = 256
    t_batch_size = min(128, t_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t = torch.linspace(0.0, 1.0, t_size, device=device)
    x0 = torch.zeros(batch_size, state_size, device=device)

    sde = SDE().to(device)
    penalty = MmdPenalty().to(device)
    optimizer = optim.Adam(sde.parameters(), lr=1e-3)
    weight = 0.5 * 1e-2

    with tqdm(range(epoch)) as pbar:
        for ep in pbar:
            optimizer.zero_grad()

            x0_big = torch.zeros(2 * batch_size, state_size, device=device)
            x_big = torchsde.sdeint(sde, x0_big, t, method='euler')  # shape: (t_size, 2*batch, state)

            x_a = x_big[:, :batch_size, :]
            x_b = x_big[:, batch_size:, :]

            idx = torch.randperm(t_size, device=device)[:t_batch_size]

            t_sel = t[idx]
            x_sel = x_a[idx]  
            x_flat = x_sel.reshape(-1, state_size)
            t_flat = t_sel.unsqueeze(1).repeat(1, batch_size).reshape(-1, 1)

            u_flat = sde.f(t_flat, x_flat)
            loss = torch.mean(u_flat ** 2) * weight

            loss += penalty(x_a[-1], x_b[-1])

            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})

    # --- サンプル生成 & 可視化 ---
    x0p = torch.zeros(200000, state_size, device=device)
    x0p_big = torch.zeros(2 * 200000, state_size, device=device)
    with torch.no_grad():
        xp_big = torchsde.sdeint(sde, x0p_big, t, method='euler')
    xp = xp_big[:, :200000, :].cpu().numpy()

    plot(xp[0], xp[84], xp[170], xp[-1])


if __name__ == "__main__":
    main()
