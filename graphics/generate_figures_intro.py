import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import patches

import common

PART_NUM = 0


def generate_fig1():

    def arrow(start, end):
        return common.arrow_by_start_end(start, end,
            head_width=0.01,
            head_length=0.01,
            length_includes_head=True,
            alpha=0.5,
            color="black")
            
    fig = Figure(figsize=(5, 5))
    canvas = FigureCanvas(fig)
    ax = fig.add_axes((0.01, 0.01, 0.98, 0.98))
    common.set_ax_params(ax)

    box_width = 0.05
    x_middle = np.linspace(0.1, 0.9, 5)
    x_start = x_middle - box_width / 2
    x_end = x_middle + box_width / 2
    heights = np.array([0.2, 0.8, 0.8, 0.8, 0.1])
    y_start = 0.5 - 0.5 * heights
    y_end = 0.5 + 0.5 * heights
    labels = [
        "state ($\mathbb{R}^{8}$)",
        "hidden$_1$ ($\mathbb{R}^{50}$, ReLU)",
        "hidden$_2$ ($\mathbb{R}^{50}$, ReLU)",
        "hidden$_3$ ($\mathbb{R}^{50}$, ReLU)",
        "output"]
    
    
    texts = []
    boxes = []
    for i in range(5):
        boxes.append(patches.FancyBboxPatch(
            (x_start[i], y_start[i]),
            width=box_width,
            height=heights[i],
            boxstyle="round,pad=0.01",
            facecolor="white"
        ))
        texts.append(ax.text(x_middle[i], 0.5, labels[i],
            ha="center", va="center", size="large", rotation=90))
    
    arrows = []
    for i in range(4):
        arrows.append(arrow(
            (x_end[i] + 0.01, y_end[i] + 0.01),
            (x_start[i+1] - 0.01, y_end[i+1])
        ))
        
        arrows.append(arrow(
            (x_end[i] + 0.01, y_end[i] + 0.01),
            (x_start[i+1] - 0.01, y_start[i+1])
        ))
        
        arrows.append(arrow(
            (x_end[i] + 0.01, y_start[i] - 0.01),
            (x_start[i+1] - 0.01, y_end[i+1])
        ))
        
        arrows.append(arrow(
            (x_end[i] + 0.01, y_start[i] - 0.01),
            (x_start[i+1] - 0.01, y_start[i+1])
        ))

    for p in boxes + arrows:
        ax.add_patch(p)
    
    common.save_next_fig(PART_NUM, fig)


if __name__ == "__main__":
    generate_fig1()
