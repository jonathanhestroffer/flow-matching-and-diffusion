import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class MLP(nn.Module):
    """
    MLP parametrization of flow and/or noise model.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], axis=1))
    
class Trainer:
    def __init__(
        self, 
        model: torch.nn.Module, 
        dist0: torch.distributions, 
        dist1: torch.distributions,
        device: torch.device,
    ):
        super().__init__()
        self.model = model
        self.dist0 = dist0
        self.dist1 = dist1
        self.device = device

    def get_optimizer(self, lr: float):
        return torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def get_batch(self, batch_size: int):
        """
        Sample time (t), noise (epsilon or x0), and z
        """
        t = torch.rand((batch_size, 1)).to(self.device)
        e = self.dist0.sample((batch_size, 1)).to(self.device)
        z = self.dist1.sample((batch_size, 1)).to(self.device)

        # interpolate
        xt = t*z + (1-t)*e
        
        return t, e, xt, z

    def train(self, num_epochs: int, lr: float = 1e-3, batch_size = 1000):
        # Start
        self.model.to(self.device)
        opt = self.get_optimizer(lr)
        self.model.train()

        # Train loop
        pbar = tqdm(enumerate(range(num_epochs)))
        for idx, epoch in pbar:
            opt.zero_grad()
            loss = self.get_train_loss(batch_size)
            loss.backward()
            opt.step()
            if idx % 100 == 0:
                pbar.set_description(f'Epoch {idx}, loss: {loss.item()}')

        # Finish
        self.model.eval()


class FlowTrainer(Trainer):
    """
    Train Flow Matching Model: u_t^theta(x)
    """
    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__(model, **kwargs)

    def get_train_loss(self, batch_size: int) -> torch.Tensor:
        # sample batch
        t, e, xt, z = self.get_batch(batch_size)

        # regress vector field
        return F.mse_loss(self.model(xt, t), z-e)

class NoiseTrainer(Trainer):
    """
    Train Noise Model: where, e_t^theta(x) = -beta*s_t^theta(x)
    """
    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__(model, **kwargs)

    def get_train_loss(self, batch_size: int):
        # sample batch
        t, e, xt, _ = self.get_batch(batch_size)

        # regress noise
        return F.mse_loss(self.model(xt, t), e)