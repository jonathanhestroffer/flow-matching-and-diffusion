from typing import List

import torch
import numpy as np

def compute_quantile(dist: torch.distributions, p: float, x_range=(-10, 10), resolution=1e-4) -> float:
    """
    Approximate the quantile at probability = `p` of distribution
    """
    x = torch.arange(*x_range, resolution)
    cdf_vals = dist.cdf(x)
    return np.interp(p, cdf_vals.numpy(), x.numpy())    

def SimulateSDE(
        flow_model: torch.nn.Module, 
        noise_model: torch.nn.Module, 
        x0: float, 
        sigma: float, 
        num_steps: int,
        device: torch.device
    ) -> List[float]:
    """
    Simulate stochastic differential equation (SDE) via Langevin dynamics 
    
    Note: 
    -> reversed time sampling convention (correct/push back added noise)
    -> (-1/beta_t) * noise_t^theta(x) = s_t^theta(x)

    Args:
        flow_model  (MLP)   : Learned flow matching model
        noise_model (MLP)   : Learned noise model
        x0          (float) : Initial point drawn from noise prior
        sigma       (float) : Diffusion coefficient
        num_steps   (int)   : Number of timesteps

    Returns:
        List[float] : Simulated SDE path
    """
    dt = 1.0 / num_steps
    x_path = []
    x = torch.tensor(x0, device=device).view(-1,1)

    for i in range(num_steps):
        noise = torch.randn(1, device=device).view(-1,1)
        t = torch.tensor(i*dt, device=device).view(-1,1)
        beta = 1-t
        # x = x + flow_model(x, t)*dt + 0.5*(sigma**2)*(-1/beta)*noise_model(x, t) + sigma*np.sqrt(dt)*noise
        x = x + (flow_model(x, t) + 0.5*(sigma**2)*(-1/beta)*noise_model(x, t))*dt + sigma*np.sqrt(dt)*noise
        x_path.append(x.item())
    return x_path

def SimulateODE(
        flow_model: torch.nn.Module, 
        x0: float, 
        num_steps: int,
        device: torch.device
    ) -> List[float]:
    """
    Simulate flow matching ordinary differential equation (ODE) via Euler Method

    Args:
        flow_model  (MLP)   : MLP flow model
        x0          (float) : Initial point drawn from noise prior
        num_steps   (int)   : Number of timesteps

    Returns:
        List[float] : Simulated ODE path
    """
    dt = 1.0 / num_steps
    x_path = []
    x = torch.tensor(x0, device=device).view(-1,1)

    for i in range(num_steps):
        t = torch.tensor(i*dt, device=device).view(-1,1)
        x = x + flow_model(x, t)*dt
        x_path.append(x.item())
    return x_path