import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import patches

import common

PART_NUM = 3


def generate_fig1():
    fig = Figure(figsize=(4, 4))
    canvas = FigureCanvas(fig)
    ax = fig.add_axes((0.01, 0.01, 0.98, 0.98))
    common.set_ax_params(ax)
    ax.axis([0., 1., 0., 1.])
    
    for i, w in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
        ax.add_patch(patches.FancyBboxPatch(
            (0.5 - w / 2., 0.5 - w / 2.),
            width=w,
            height=w,
            boxstyle="round,pad=0.01",
            facecolor="blue",
            edgecolor="None",
            alpha=0.3
        ))
    
    for i, w in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
        ax.add_patch(patches.FancyBboxPatch(
            (0.5 - w / 2., 0.5 - w / 2.),
            width=w,
            height=w,
            boxstyle="round,pad=0.01",
            facecolor="None",
            edgecolor="black",
        ))
        
        if i > 0:
            ax.text(0.5 - w / 2., 0.5 - w / 2.,
                "$\\pi_{}=$MCTS$(\\pi_{})$".format(i, i-1),
                ha="left", va="bottom", size="small", color="yellow")
        else:
            ax.text(0.5 - w / 2., 0.5 - w / 2.,
                "$\\pi_0$", ha="left", va="bottom", size="small",
                color="yellow")
    
    common.save_next_fig(PART_NUM, fig)


def generate_fig2():
    fig = Figure(figsize=(4, 10))
    canvas = FigureCanvas(fig)
    ax = fig.add_axes((0.01, 0.01, 0.98, 0.98))
    common.set_ax_params(ax)
    ax.axis([0., 1., 0., 2.5])
    
    w0 = 0.1
    w1 = 0.3
    
    H = np.linspace(2.2, 0.3, 5)
    
    for i, h  in enumerate(H):
        for w in [w0, w1]:
            ax.add_patch(patches.FancyBboxPatch(
                (0.5 - w / 2., h - w / 2.),
                width=w,
                height=w,
                boxstyle="round,pad=0.01",
                facecolor="blue",
                edgecolor="None",
                alpha=0.3
            ))
        for w in [w0, w1]:
            ax.add_patch(patches.FancyBboxPatch(
                (0.5 - w / 2., h - w / 2.),
                width=w,
                height=w,
                boxstyle="round,pad=0.01",
                facecolor="None",
                edgecolor="black"
            ))
        ax.text(0.5 - w0 / 2., h - w0 / 2.,
            "$\\pi_{}$".format(i),
            ha="left", va="bottom", size="large", color="yellow")            
        ax.text(0.5 - w1 / 2., h - w1 / 2.,
            "MCTS$(\\pi_{})$".format(i),
            ha="left", va="bottom", size="large", color="yellow")
        
        if i < 4:
            ax.add_artist(common.arrow_by_start_end(
                [0.5 - w1 / 2. - 0.01, h - w1 / 2. - 0.01],
                [0.5 - w0 / 2. - 0.01, H[i + 1] + w0 / 2. + 0.01],
                color="black", width=0.005,
                length_includes_head=True,
                alpha=0.3))
        
            ax.add_artist(common.arrow_by_start_end(
                [0.5 + w1 / 2. + 0.01, h - w1 / 2. - 0.01],
                [0.5 + w0 / 2. + 0.01, H[i + 1] + w0 / 2. + 0.01],
                color="black", width=0.005,
                length_includes_head=True,
                alpha=0.3))

    common.save_next_fig(PART_NUM, fig)


DATA_FILES = [
    common.DataFile("alpha_pong", "Alpha Pong Network", 100, []),
    common.DataFile("alpha_pong_mcts", "MCTS(Alpha Pong Network)", 100, []),
    common.DataFile("alpha_pong_self", "Alpha Pong Network", 100, []),
    common.DataFile("alpha_pong_self_mcts", "MCTS(Alpha Pong Network)", 100, []),
]

if __name__ == "__main__":
    generate_fig1()
    generate_fig2()
    common.fill_data(DATA_FILES)
    common.compare_plots(DATA_FILES[:2], PART_NUM,
        title="Win Rate of Alpha-Pong-Zero", show_time=True)
    common.compare_plots(DATA_FILES[2:], PART_NUM,
        title="Win Rate of Alpha-Pong-Zero, trained using self-play", show_time=True)
