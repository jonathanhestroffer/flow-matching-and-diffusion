import torch
from torch.distributions import Normal, Categorical, MixtureSameFamily

from model import MLP, FlowTrainer, NoiseTrainer
from utils import compute_quantile, SimulateODE, SimulateSDE
from plotting import plot_trajectories

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # Create p_init as a Gaussian
    mean    = 0
    sigma   = 1
    p_init  = Normal(loc=mean, scale=sigma)

    # Create p_data as a bimodal Gaussian
    means   = torch.tensor([-3.0, 3.0])
    sigmas  = torch.tensor([1.0, 1.0])
    probs   = torch.tensor([0.5, 0.5])

    component_distribution  = Normal(loc=means, scale=sigmas)
    mixing_distribution     = Categorical(probs=probs)

    p_data = MixtureSameFamily(mixing_distribution, component_distribution)

    # Flow Matching model
    # - flow model learns vector field u_t^{\theta}(x)
    trainer = FlowTrainer(model=MLP(), dist0=p_init, dist1=p_data, device=device)
    trainer.train(num_epochs=4000, lr=1e-3, batch_size=1000)
    flow_model = trainer.model

    # Diffusion model
    # - diffusion model learns noise e_t^{\theta}(x)
    #   instead of score for better training stability
    # Train Flow Matching Model
    trainer = NoiseTrainer(model=MLP(), dist0=p_init, dist1=p_data, device=device)
    trainer.train(num_epochs=4000, lr=1e-3, batch_size=1000)
    noise_model = trainer.model

    # Get ODE and SDE paths sampled at different quantiles of p_init
    ODE_trajectories, SDE_trajectories = [], []
    num_steps, sigma = 50, 0.5
    for q in torch.arange(0.01, 1, 0.01):
        quantile = compute_quantile(p_init, q).item()
        ODE_trajectories.append(SimulateODE(flow_model, quantile, num_steps, device=device))
        SDE_trajectories.append(SimulateSDE(flow_model, noise_model, quantile, sigma, num_steps, device=device))

    # Create and save figures
    fig = plot_trajectories(p_init, p_data, ODE_trajectories, "b", 0.3)
    fig.suptitle(r"Flow Matching Trajectories, model: $u_t^{\theta}(x)$")
    fig.savefig("./flow.png",dpi=300)

    fig = plot_trajectories(p_init, p_data, SDE_trajectories, "g", 0.3)
    fig.suptitle(r"Denoising Diffusion Trajectories, model: $\epsilon_t^{\theta}(x)=-\beta_t s_t^{\theta}(x)$")
    fig.savefig("./diffusion.png",dpi=300)


