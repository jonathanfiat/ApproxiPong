from itertools import product

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import patches

import common

PART_NUM = 4


def generate_fig1():
    fig = Figure(figsize=(8, 4))
    canvas = FigureCanvas(fig)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    for ax in [ax1, ax2]:
        common.set_ax_params(ax)
        ax.axis([0., 1., 0., 1.])

    ax1.set_title("Sequential Learning")
    ax2.set_title("Parallel Learning")

    box_width = 0.1
    box_height = 0.1
    y_middle = np.linspace(0.9, 0.1, 5)
    y_start = y_middle - box_height / 2
    y_end = y_middle + box_height / 2

    x_start = 0.5 - 0.5 * box_width
    x_middle = 0.5
    x_end = 0.5 + 0.5 * box_width
    
    for i in range(0, 5, 2):
        ax1.add_patch(patches.FancyBboxPatch(
            (x_start, y_start[i]),
            width=box_width,
            height=box_height,
            boxstyle="round,pad=0.01",
            facecolor="lightblue"
        ))
        ax1.text(x_middle, y_middle[i], "$\\pi_{{{}}}$".format(i // 2),
            ha="center", va="center", size="large")
        
    for i in range(1, 5, 2):
        ax1.add_patch(patches.FancyBboxPatch(
            (x_start, y_start[i]),
            width=box_width,
            height=box_height,
            boxstyle="round,pad=0.01",
            facecolor="lightgreen"
        ))
        ax1.text(x_middle, y_middle[i], "$e_{{{}}}$".format(i // 2),
            ha="center", va="center", size="large")
        ax1.add_patch(common.arrow_by_start_end(
            (x_middle, y_start[i - 1] - 0.01),
            (x_middle, y_end[i] + 0.01),
            width=0.005,
            length_includes_head=True,
            color="black"
        ))
        ax1.add_patch(common.arrow_by_start_end(
            (x_middle, y_start[i] - 0.01),
            (x_middle, y_end[i + 1] + 0.01),
            width=0.005,
            length_includes_head=True,
            color="black"
        ))

    x2_middle = np.linspace(0.9, 0.1, 5)
    x2_start = y_middle - box_width / 2
    x2_end = y_middle + box_width / 2
    
    for i in range(0, 5, 2):
        ax2.add_patch(patches.FancyBboxPatch(
            (x_start, y_start[i]),
            width=box_width,
            height=box_height,
            boxstyle="round,pad=0.01",
            facecolor="lightblue"
        ))
        ax2.text(x_middle, y_middle[i], "$\\pi_{{{}}}$".format(i // 2),
            ha="center", va="center", size="large")

    for i in range(1, 5, 2):
        for j in range(0, 5):
            ax2.add_patch(patches.FancyBboxPatch(
                (x2_start[j], y_start[i]),
                width=box_width,
                height=box_height,
                boxstyle="round,pad=0.01",
                facecolor="lightgreen"
            ))
            ax2.text(x2_middle[j], y_middle[i], "$e^{{({})}}_{{{}}}$".format(i // 2, 5-j),
                ha="center", va="center", size="large")
            ax2.add_patch(common.arrow_by_start_end(
                (x_middle, y_start[i - 1] - 0.01),
                (x2_middle[j], y_end[i] + 0.01),
                width=0.005,
                length_includes_head=True,
                color="black"
            ))
            ax2.add_patch(common.arrow_by_start_end(
                (x2_middle[j], y_start[i] - 0.01),
                (x_middle, y_end[i + 1] + 0.01),
                width=0.005,
                length_includes_head=True,
                color="black"
            ))

    common.save_next_fig(PART_NUM, fig)


def generate_fig2():
    
    for i, s, c, a, l in [
        (10, 0, "red", 1., "Failed Episode"),
        (11, 180, "purple", 0.5, "Negative Failed Episode")]:

        fig = Figure(figsize=(4, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)    
        common.set_ax_params(ax)
        ax.axis([0., 1., 0., 1.])
        ax.set_title("Policies Space")

        ax.add_artist(patches.Wedge((0.5, 0.5), 1., -30, 30, facecolor="blue",
            alpha=0.2, edgecolor="None"))
        ax.add_artist(patches.Wedge((0.5, 0.5), 1., 30, -30, facecolor="red",
            alpha=0.2, edgecolor="None"))
        
        ax.add_artist(common.arrow_by_start_size_angle((0.5, 0.5), 0.4,
            np.radians(20), width=0.01, length_includes_head=True,
            color="blue"))
        ax.add_artist(common.arrow_by_start_size_angle((0.5, 0.5), 0.4,
            np.radians(-5), width=0.01, length_includes_head=True,
            color="blue"))
        ax.add_artist(common.arrow_by_start_size_angle((0.5, 0.5), 0.4,
            np.radians(s + 50), width=0.01, length_includes_head=True, color=c,
            alpha=a))
        ax.add_artist(common.arrow_by_start_size_angle((0.5, 0.5), 0.4,
            np.radians(s + 110), width=0.01, length_includes_head=True, color=c,
            alpha=a))
        ax.add_artist(common.arrow_by_start_size_angle((0.5, 0.5), 0.4,
            np.radians(s + 170), width=0.01, length_includes_head=True, color=c,
            alpha=a))
        ax.add_artist(common.arrow_by_start_size_angle((0.5, 0.5), 0.4,
            np.radians(s + 240), width=0.01, length_includes_head=True, color=c,
            alpha=a))
        ax.add_artist(common.arrow_by_start_size_angle((0.5, 0.5), 0.4,
            np.radians(s + 280), width=0.01, length_includes_head=True, color=c,
            alpha=a))
        
        ax.add_artist(patches.Circle((0.5, 0.5), radius=0.02, color="black"))
        
        ax.legend(handles=[
                patches.Patch(color="blue", label="Better Policies",
                    alpha=0.2),
                patches.Patch(color="red", label="Worse Policies", alpha=0.2),
                patches.FancyArrow(0., 0., 1., 1., color="blue",
                    label="Successful Episode"),
                patches.FancyArrow(0., 0., 1., 1., color=c, alpha=a, label=l),
                patches.Circle((0.5, 0.5), radius=0.02, color="black",
                    label="Current Policy")
            ],
            edgecolor="black",
            loc="upper left",
            fontsize="x-small",
        )
            
        common.save_next_fig(PART_NUM, fig)


def generate_fig3():
    
    for i, s, c, a, l in [
        (12, 0, "red", 1., "Failed Episode"),
        (13, 180, "purple", 0.5, "Negative Failed Episode")]:

        fig = Figure(figsize=(4, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)    
        common.set_ax_params(ax)
        ax.axis([0., 1., 0., 1.])
        ax.set_title("Policies Space")

        ax.add_artist(patches.Wedge((0.5, 0.5), 1., -30, 30, facecolor="blue",
            alpha=0.2, edgecolor="None"))
        ax.add_artist(patches.Wedge((0.5, 0.5), 1., 30, -30, facecolor="red",
            alpha=0.2, edgecolor="None"))
        
        ax.add_artist(common.arrow_by_start_size_angle((0.5, 0.5), 0.4,
            np.radians(-25), width=0.01, length_includes_head=True,
            color="blue"))
        ax.add_artist(common.arrow_by_start_size_angle((0.5, 0.5), 0.4,
            np.radians(10), width=0.01, length_includes_head=True,
            color="blue"))
        ax.add_artist(common.arrow_by_start_size_angle((0.5, 0.5), 0.4,
            np.radians(s + 40), width=0.01, length_includes_head=True, color=c,
            alpha=a))
        ax.add_artist(common.arrow_by_start_size_angle((0.5, 0.5), 0.4,
            np.radians(s + 55), width=0.01, length_includes_head=True, color=c,
            alpha=a))
        ax.add_artist(common.arrow_by_start_size_angle((0.5, 0.5), 0.4,
            np.radians(s -50), width=0.01, length_includes_head=True, color=c,
            alpha=a))
        
        ax.add_artist(patches.Circle((0.5, 0.5), radius=0.02, color="black"))
        
        ax.legend(handles=[
                patches.Patch(color="blue", label="Better Policies", alpha=0.2),
                patches.Patch(color="red", label="Worse Policies", alpha=0.2),
                patches.FancyArrow(0., 0., 1., 1., color="blue",
                    label="Successful Episode"),
                patches.FancyArrow(0., 0., 1., 1., color=c, alpha=a, label=l),
                patches.Circle((0.5, 0.5), radius=0.02, color="black",
                    label="Current Policy")
            ],
            edgecolor="black",
            loc="upper left",
            fontsize="x-small",
        )
            
        common.save_next_fig(PART_NUM, fig)


DATA_FILES = [
    common.DataFile("policy_gradient", "Policy Gradient", 1000, []),
    common.DataFile("policy_gradient_self", "Policy Gradient\nSelf-Play Mode", 1000, []),
    common.DataFile("success_learning", "Success Learning", 1000, []),
    common.DataFile("success_learning_self", "Success Learning\nSelf-Play Mode", 1000, []),
]


if __name__ == "__main__":
    generate_fig1()
    generate_fig2()
    generate_fig3()
    common.generate_plots(DATA_FILES, PART_NUM)
