from itertools import product

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import patches, lines

import common

PART_NUM = 6


def generate_fig1():
    fig = Figure(figsize=(8, 4))
    canvas = FigureCanvas(fig)
    ax = fig.add_axes((0.01, 0.01, 0.98, 0.98))
    common.set_ax_params(ax)
    ax.axis([0., 2., 0., 1.])
    
    w = 0.3
    h = 0.2
    for x,y in product([0.3, 1., 1.7], [0.3, 0.7]):
        ax.add_patch(patches.FancyBboxPatch(
            (x - w / 2., y - h / 2.),
            width=w,
            height=h,
            boxstyle="round,pad=0.01",
            facecolor="lightblue",
            edgecolor="darkblue"
        ))

    ax.text(0.3, 0.7, "Q-iteration", ha="center", va="center", size="large")
    ax.text(0.3, 0.3, "P-iteration", ha="center", va="center", size="large")
    ax.text(1., 0.7, "Q-learning", ha="center", va="center", size="large")
    ax.text(1., 0.3, "P-learning", ha="center", va="center", size="large")
    ax.text(1.7, 0.7, "Deep\nQ-learning", ha="center", va="center",
        size="large")
    ax.text(1.7, 0.3, "Deep\nP-learning", ha="center", va="center",
        size="large")

    w = 0.25
    h = 0.1
    for x,y in product([1.3/2., 2.7/2.], [0.3, 0.7]):
        ax.add_patch(patches.FancyBboxPatch(
            (x - w / 2. - 0.02, y - h / 2.),
            width=w,
            height=h,
            boxstyle="rarrow,pad=0.01",
            facecolor="lightgreen",
            edgecolor="darkgreen"
        ))
    
    ax.text(1.3/2., 0.7, "Unknown\nModel", ha="center", va="center",
        size="small")
    ax.text(1.3/2., 0.3, "Unknown\nModel", ha="center", va="center",
        size="small")
    ax.text(2.7/2., 0.7, "Continuous\nState Space", ha="center", va="center",
        size="small")
    ax.text(2.7/2., 0.3, "Continuous\nState Space", ha="center", va="center",
        size="small")

    common.save_next_fig(PART_NUM, fig)


def generate_fig2():
    fig = Figure(figsize=(4, 4))
    canvas = FigureCanvas(fig)
    ax = fig.add_axes((0.01, 0.01, 0.98, 0.98))
    common.set_ax_params(ax)
    ax.set_facecolor("white")
    ax.axis([0., 1., 0., 1.])
    
    ax.add_artist(patches.Rectangle((0.2, 0.1), 0.7, 0.7,
        facecolor="lightgrey", edgecolor="None"))
    
    ax.add_artist(lines.Line2D([0.55, 0.55], [0.1, 0.8], color="red"))
    ax.add_artist(lines.Line2D([0.2, 0.2], [0.1, 0.8], color="black"))
    ax.add_artist(lines.Line2D([0.9, 0.9], [0.1, 0.8], color="black"))
    
    ax.add_artist(lines.Line2D([0.2, 0.9], [0.1, 0.1], color="black"))
    ax.add_artist(lines.Line2D([0.2, 0.9], [0.8, 0.8], color="black"))
    ax.add_artist(lines.Line2D([0.2, 0.9], [0.45, 0.45], color="black"))
    
    ax.text((0.2 + 0.55) / 2., 0.87, "Known\nModel", ha="center", va="center",
        size="medium")
    ax.text((0.9 + 0.55) / 2., 0.87, "Unknown\nModel", ha="center", va="center",
        size="medium")
    
    ax.text(0.1, (0.1 + 0.45) / 2., "Continuous", ha="center", va="center",
        size="medium", rotation=50)
    ax.text(0.1, (0.8 + 0.45) / 2., "Discrete", ha="center", va="center",
        size="medium", rotation=50)
    
    ax.text((0.2 + 0.55) / 2., (0.1 + 0.45) / 2.,
        "Deep-Q-Iteration\n\nDeep-P-Iteration",
        ha="center", va="center", size="small")
    ax.text((0.9 + 0.55) / 2., (0.1 + 0.45) / 2.,
        "Deep-Q-Learning\n\nDeep-P-Learning",
        ha="center", va="center", size="small")
    
    ax.text((0.2 + 0.55) / 2., (0.8 + 0.45) / 2.,
        "Q-Iteration\n\nP-Iteration",
        ha="center", va="center", size="small")
    ax.text((0.9 + 0.55) / 2., (0.8 + 0.45) / 2.,
        "Q-Learning\n\nP-Learning",
        ha="center", va="center", size="small")
    
    common.save_next_fig(PART_NUM, fig)


DATA_FILES = [
    common.DataFile("actor_critic", "Actor Critic", 1000, []),
    common.DataFile("actor_critic_self", "Actor Critic\nSelf-Play Mode", 1000, []),
    common.DataFile("success_learning_critic", "Success Learning with a Critic", 1000, []),
    common.DataFile("success_learning_critic_self", "Success Learning with a Critic\nSelf-Play Mode", 1000, []),
    common.DataFile("deep_p", "Deep-P-Learning", 1000, []),
    common.DataFile("deep_p_self", "Deep-P-Learning\nSelf-Play Mode", 1000, [])
]


if __name__ == "__main__":
    generate_fig1()
    generate_fig2()
    common.generate_plots(DATA_FILES, PART_NUM)
