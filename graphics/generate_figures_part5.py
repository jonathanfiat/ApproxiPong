from itertools import product

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import patches, lines, gridspec

import common

PART_NUM = 5


ELLIPSE_SMALL = [
    ((0.5, 0.5), 0.30, 0.30, 0), 
    ((0.55, 0.52), 0.45, 0.25, 20),
    ((0.6, 0.55), 0.55, 0.15, 30),
    ((0.63, 0.58), 0.60, 0.10, 40)
]


ELLIPSE_BIG = [
    ((0.5, 0.5), 0.31, 0.31, 0), 
    ((0.55, 0.52), 0.46, 0.26, 20),
    ((0.6, 0.55), 0.56, 0.16, 30),
    ((0.63, 0.58), 0.61, 0.11, 40)
]

ELLIPSE_SMALL_ALT = ((0.55, 0.52), 0.60, 0.20, 20)
ELLIPSE_BIG_ALT = ((0.55, 0.52), 0.61, 0.21, 20)


def generate_fig1():
    fig = Figure(figsize=(5, 5))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)    
    common.set_ax_params(ax)
    ax.axis([0., 1., 0., 1.])
    ax.set_title("State Space")
    
    e = patches.Ellipse(*ELLIPSE_SMALL[0], edgecolor="darkgreen",
        facecolor="green", linewidth=1, alpha=0.4, label="Possible $\\pi_0$ States")
    ax.add_artist(e)
    x = np.random.random(500)
    y = np.random.random(500)
    s = lines.Line2D(x, y, linestyle="None", marker=".", label="Samples",
        markersize=2, color="darkblue")
    ax.add_artist(s)
    s.set_clip_path(e)
    
    ax.legend(handles=[e, s],
        edgecolor="black",
        loc="upper left",
        fontsize="x-small")
    
    common.save_next_fig(PART_NUM, fig)


def generate_fig2():
    fig = Figure(figsize=(5, 5))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)    
    common.set_ax_params(ax)
    ax.axis([0., 1., 0., 1.])
    ax.set_title("State Space")
    
    e = patches.Ellipse(*ELLIPSE_BIG[0], edgecolor="None",
        facecolor="blue", linewidth=1, alpha=0.2, label="1 Q update")
    ax.add_artist(e)  
    
    ax.legend(handles=[e],
        edgecolor="black",
        loc="upper left",
        fontsize="x-small")

    common.save_next_fig(PART_NUM, fig)


def generate_fig3():
    fig = Figure(figsize=(5, 5))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)    
    common.set_ax_params(ax)
    ax.axis([0., 1., 0., 1.])
    ax.set_title("State Space")

    e = patches.Ellipse(*ELLIPSE_SMALL[1], edgecolor="darkgreen",
        facecolor="green", linewidth=1, alpha=0.4, label="Possible $\\pi_1$ States")
    ax.add_artist(e)
    x = np.random.random(500)
    y = np.random.random(500)
    s = lines.Line2D(x, y, linestyle="None", marker=".", label="Samples",
        markersize=2, color="darkblue")
    ax.add_artist(s)
    s.set_clip_path(e)

    ax.legend(handles=[e, s],
        edgecolor="black",
        loc="upper left",
        fontsize="x-small")

    common.save_next_fig(PART_NUM, fig)


def generate_fig4():
    fig = Figure(figsize=(5, 5))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)    
    common.set_ax_params(ax)
    ax.axis([0., 1., 0., 1.])
    ax.set_title("State Space")
    
    ax.add_artist(patches.Ellipse(*ELLIPSE_BIG[0], edgecolor="None",
        facecolor="blue", linewidth=1, alpha=0.2))
        
    ax.add_artist(patches.Ellipse(*ELLIPSE_BIG[1], edgecolor="None",
        facecolor="blue", linewidth=1, alpha=0.2))
    
    e_0 = patches.Ellipse(*ELLIPSE_BIG[0], edgecolor="None",
        facecolor="blue", linewidth=1, alpha=1 - 0.8,
        label="1 Q update")
    e_1 = patches.Ellipse(*ELLIPSE_BIG[0], edgecolor="None",
        facecolor="blue", linewidth=1, alpha=1 - 0.8**2,
        label="2 Q updates")

    ax.legend(handles=[e_0, e_1],
        edgecolor="black",
        loc="upper left",
        fontsize="x-small")
    
    common.save_next_fig(PART_NUM, fig)


def generate_fig5():
    fig = Figure(figsize=(5, 5))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)    
    common.set_ax_params(ax)
    ax.axis([0., 1., 0., 1.])
    ax.set_title("State Space")
    
    e = patches.Ellipse(*ELLIPSE_SMALL[3], edgecolor="darkgreen",
        facecolor="green", linewidth=1, alpha=0.4, label="Possible $\\pi_n$ States")
    ax.add_artist(e)
    ax.legend(handles=[e],
        edgecolor="black",
        loc="upper left",
        fontsize="x-small")
        
    common.save_next_fig(PART_NUM, fig)

    
def generate_fig6():
    fig = Figure(figsize=(5, 5))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)    
    common.set_ax_params(ax)
    ax.axis([0., 1., 0., 1.])
    ax.set_title("State Space")
    
    handles = []
    
    for i, p in enumerate(ELLIPSE_BIG):
        ax.add_artist(patches.Ellipse(*p,
            edgecolor="None", facecolor="blue", linewidth=1, alpha=0.2))
        handles.append(patches.Ellipse((0., 0.), 0., 0., 0, edgecolor="None",
            facecolor="blue", linewidth=1, alpha=1-0.8**(i+1),
            label="{} Q update{}".format(i+1, "s" if i else "")))
    
    
    ax.legend(handles=handles,
        edgecolor="black",
        loc="upper left",
        fontsize="x-small")

    common.save_next_fig(PART_NUM, fig)


def generate_fig7():
    fig = Figure(figsize=(5, 5))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)    
    common.set_ax_params(ax)
    ax.axis([0., 1., 0., 1.])
    ax.set_title("State Space")
    
    color_ellipses = []
    bg_ellipses = []
    for i, p in enumerate(ELLIPSE_BIG):
        color_ellipses.append(patches.Ellipse(*p,
            edgecolor="None", facecolor="blue", linewidth=1,
            alpha=1-0.8**(4-i)))
        bg_ellipses.append(patches.Ellipse(*p,
            edgecolor="None", facecolor="lightgrey", linewidth=1))
    
    for i in range(3, -1, -1):
        ax.add_artist(color_ellipses[i])
        if i:
            ax.add_artist(bg_ellipses[i-1])
            bg_ellipses[i].set_clip_path(color_ellipses[i])
    
    handles = []
    for i in range(4):
        handles.append(patches.Ellipse((0., 0.), 0., 0., 0, edgecolor="None",
            facecolor="blue", linewidth=1, alpha=1-0.8**(i+1),
            label="{} Q update{}".format(i+1, "s" if i else "")))
    
    ax.legend(handles=handles,
        edgecolor="black",
        loc="upper left",
        fontsize="x-small")
    
    common.save_next_fig(PART_NUM, fig)


def generate_fig8(width=8):
    fig = Figure(figsize=(width, 8))
    canvas = FigureCanvas(fig)
    gs = gridspec.GridSpec(4, 4)
    fig.suptitle("Visual Comparison of *-Q-* Algorithms")
    
    tabl_q_itert = []
    deep_q_itert = []
    deep_q_learn = []
    deep_q_lr_re = []
    
    for i in range(4):
        tabl_q_itert.append(fig.add_subplot(gs[i, 0]))
        deep_q_itert.append(fig.add_subplot(gs[i, 1]))
        deep_q_learn.append(fig.add_subplot(gs[i, 2]))
        deep_q_lr_re.append(fig.add_subplot(gs[i, 3]))
    
    tabl_q_itert[0].set_title("Tabular Q-Iteration")
    deep_q_itert[0].set_title("Deep-Q-Iteration")
    deep_q_learn[0].set_title("Deep-Q-Learning")
    deep_q_lr_re[0].set_title("Deep-Q-Learning\nwith Replay DB")
    
    for ax in tabl_q_itert + deep_q_itert + deep_q_learn + deep_q_lr_re:
        common.set_ax_params(ax)
        ax.axis([0., 1., 0., 1.])

    for ax in tabl_q_itert:
        for v in np.linspace(0, 1, 6)[1:-1]:
            ax.add_artist(lines.Line2D([0., 1.], [v, v], color="black"))
            ax.add_artist(lines.Line2D([v, v], [0., 1.], color="black"))
    
    for ax in tabl_q_itert[::2]:
        X = np.linspace(0.1, 0.9, 5)
        Y = np.linspace(0.1, 0.9, 5)
        x, y = zip(*product(X, Y))
        ax.add_artist(lines.Line2D(x, y, linestyle="None", marker=".",
            markersize=2, color="darkblue"))
    
    
    for i, ax in enumerate(tabl_q_itert[1::2]):
        ax.add_artist(patches.Rectangle((0.,0.), 1., 1., alpha=1-0.8**(i+1),
            facecolor="blue"))

    for ax in deep_q_itert[::2]:
        x = np.random.random(50)
        y = np.random.random(50)
        ax.add_artist(lines.Line2D(x, y, linestyle="None", marker=".",
            markersize=2, color="darkblue"))
    
    for i, ax in enumerate(deep_q_itert[1::2]):
        ax.add_artist(patches.Rectangle((0.,0.), 1., 1., alpha=1-0.8**(i+1),
            facecolor="blue"))
    
    for ax in [deep_q_learn[0], deep_q_lr_re[0]]:
        e = patches.Ellipse(*ELLIPSE_SMALL[0], edgecolor="darkgreen",
            facecolor="green", linewidth=1, alpha=0.4, label="Possible $\\pi_0$ States")
        ax.add_artist(e)
        x = np.random.random(300)
        y = np.random.random(300)
        s = lines.Line2D(x, y, linestyle="None", marker=".", label="Samples",
            markersize=2, color="darkblue")
        ax.add_artist(s)
        s.set_clip_path(e)
    
    for ax in [deep_q_learn[1], deep_q_lr_re[1]]:
        e = patches.Ellipse(*ELLIPSE_BIG[0], edgecolor="None",
            facecolor="blue", linewidth=1, alpha=0.2, label="1 Q update")
        ax.add_artist(e)
    
    for ax in [deep_q_learn[2], deep_q_lr_re[2]]:
        e = patches.Ellipse(*ELLIPSE_SMALL_ALT, edgecolor="darkgreen",
            facecolor="green", linewidth=1, alpha=0.4, label="Possible $\\pi_1$ States")
        ax.add_artist(e)
        x = np.random.random(300)
        y = np.random.random(300)
        s = lines.Line2D(x, y, linestyle="None", marker=".", label="Samples",
            markersize=2, color="darkblue")
        ax.add_artist(s)
        s.set_clip_path(e)
    
    deep_q_learn[3].add_patch(patches.Ellipse(*ELLIPSE_SMALL[0],
        edgecolor="None", facecolor="blue", linewidth=1, alpha=1-0.8,
        label="1 Q update"))
    deep_q_learn[3].add_patch(patches.Ellipse(*ELLIPSE_BIG_ALT,
        edgecolor="None", facecolor="blue", linewidth=1, alpha=1-0.8,
        label="2 Q update"))
    
    
    e1 = patches.Ellipse(*ELLIPSE_BIG[0],
        edgecolor="None", facecolor="blue", linewidth=1, alpha=1-0.8**2,
        label="1 Q update")
    e2 = patches.Ellipse(*ELLIPSE_BIG_ALT,
        edgecolor="None", facecolor="blue", linewidth=1, alpha=1-0.8,
        label="2 Q update")
    e3 = patches.Ellipse(*ELLIPSE_BIG_ALT,
        edgecolor="None", facecolor="lightgrey")
    deep_q_lr_re[3].add_patch(e2)
    deep_q_lr_re[3].add_patch(e3)
    deep_q_lr_re[3].add_patch(e1)
    e3.set_clip_path(e1)
    
    common.save_next_fig(PART_NUM, fig)


DATA_FILES = [
    common.DataFile("deep_q_deepmind", "Deep-Q-Learning\nDeepMind's version", 1, []),
    common.DataFile("deep_q_no_replay", "Deep-Q-Learning\nNo Replay DB",1000, []),
    common.DataFile("deep_q_replay", "Deep-Q-Learning\nWith Replay DB", 1000, []),
    common.DataFile("deep_q_replay_many", "Deep-Q-Learning\nWith Replay DB and 5 iterations", 1000, []),
    common.DataFile("double_deep_q", "Double Deep-Q-Learning", 1000, []),
    common.DataFile("deep_q_self", "Deep-Q-Learning\nSelf-Play Mode", 1000, []),
]


if __name__ == "__main__":
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    generate_fig6()
    generate_fig7()
    generate_fig8()
    common.generate_plots(DATA_FILES, PART_NUM)
    generate_fig8(16)
