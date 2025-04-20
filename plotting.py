import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import List, Union

def remove_spines(axs: Union[List[Axes], Axes], positions: List[str]):
    """
    Remove specified ax.spines
    """
    if not isinstance(axs, list):
        axs = [axs]
    for ax in axs:
        for pos in positions:
            ax.spines[pos].set_visible(False)

def plot_trajectories(
        p_init: torch.distributions, 
        p_data: torch.distributions, 
        trajectories: List[List[float]], 
        line_color="g", line_alpha=0.5
    ) -> Figure:

    plt.rcParams.update({"font.size":16})
    fig, axs = plt.subplots(figsize=(12,5), nrows=2, ncols=1, height_ratios=(0.3,0.7))

    # p_init and p_data PDFs
    x = torch.arange(-8, 8, 0.1)
    y_init = torch.exp(p_init.log_prob(x))
    y_data = torch.exp(p_data.log_prob(x))

    # times
    num_steps = len(trajectories[0])
    ts = torch.arange(num_steps)

    for trajectory in trajectories:
        axs[1].plot(trajectory, ts, c=line_color, alpha=line_alpha)
        axs[1].set_yticks([0, num_steps-1],[r"$X_0\sim p_{init}$",r"$X_1\sim p_{data}$"])

    axs[1].set_ylim(0, num_steps-1)

    # plot p_init
    ax1_2 = plt.twinx(ax=axs[1])
    ax1_2.plot(x, y_init, c='k')
    ax1_2.set_ylim(0, 2*ax1_2.get_ylim()[1])
    ax1_2.set_xticks([],[])
    ax1_2.set_yticks([],[])

    # plot p_data, bimodal Gaussian
    axs[0].plot(x, y_data, c='k')
    axs[0].set_ylim(0, 2*axs[0].get_ylim()[1])
    axs[0].set_xticks([],[])
    axs[0].set_yticks([],[])

    fig.text(x=0.6,y=0.4,s="")
    # cleanup
    remove_spines(axs[0], ["left","right","top"])
    remove_spines([axs[1], ax1_2], ["left","right"])

    fig.subplots_adjust(hspace=0)
    
    return fig
