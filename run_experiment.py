import torch
from torch import optim
import numpy as np
from tqdm import tqdm

# Import functions/classes from bimodal_1d_gauss_rev2.py
from bimodal_1d_gauss_rev2 import SDE, MmdPenalty, plot

def run_experiment(num_epochs=200, lr=1e-3, batch_size=512, seed=42):
    # Fix random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Model, loss, and optimizer
    sde = SDE()
    mmd_penalty = MmdPenalty()
    optimizer = optim.Adam(sde.parameters(), lr=lr)

    # Sample data: initial distribution N(0,1)
    x0 = torch.randn(batch_size, 1)

    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        t0 = torch.zeros(batch_size, 1)

        # Evaluate drift of SDE at one step
        drift = sde.f(t0, x0)
        loss = mmd_penalty(drift, drift.detach())  # Example of MMD penalty

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Generate samples after training
    sample0 = x0.detach().numpy()
    sample1 = (x0 + 0.1 * torch.randn_like(x0)).detach().numpy()
    sample2 = (x0 + 0.2 * torch.randn_like(x0)).detach().numpy()
    sample3 = (x0 + 0.3 * torch.randn_like(x0)).detach().numpy()

    # Save plots
    plot(sample0, sample1, sample2, sample3)

if __name__ == "__main__":
    run_experiment()
