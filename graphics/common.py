from collections import namedtuple
import re

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerPatch
from matplotlib.legend import Legend
from matplotlib import patches, style


TOP = 1.
BOTTOM = -1.
LEFT = -1.
RIGHT = 1.
BALL_RADIUS = 0.03
BALL_COLOR = "g"
PADDLE_WIDTH = 0.03
PADDLE_COLOR = "r"
NAME_COLOR = "lightblue"
HPL = 0.2

def arrow_by_start_end(start, end, **kwargs):
    return patches.FancyArrow(
        start[0],
        start[1],
        end[0] - start[0],
        end[1] - start[1],
        **kwargs
    )


def arrow_by_start_size_angle(start, size, angle, **kwargs):
    return patches.FancyArrow(
        start[0],
        start[1],
        size * np.cos(angle),
        size * np.sin(angle),
        **kwargs
    )


def arrow_by_start_size_dir(start, size, direction, **kwargs):
    angle = np.arctan2(*direction)
    return patches.FancyArrow(
        start[0],
        start[1],
        size * np.cos(angle),
        size * np.sin(angle),
        **kwargs
    )


class HandlerCircle(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = patches.Circle(center, radius=fontsize / 2)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


class HandlerArrow(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = patches.FancyArrow(
            xdescent, 0.5 * height,
            width, 0.,
            length_includes_head=True,
            head_width=0.75 * height,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
        )
        p.set_transform(trans)
        return [p]


Legend.update_default_handler_map({
    patches.Circle: HandlerCircle(),
    patches.FancyArrow: HandlerArrow()
})


def set_ax_params(ax, facecolor="lightgrey"):
    ax.set_aspect(1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params("both", bottom="off", left="off")
    ax.set_facecolor(facecolor)


R = re.compile(r"^(\d+)\(([\d\.]+)\) : (\d+)\|(\d+)\|(\d+)$", re.MULTILINE)
Stat = namedtuple("Stat", ["episodes", "time", "win_rate", "lose_rate", "draw_rate"])


def read_data(fn, episodes_per_1=1000):
    res = R.findall(open(fn, "r").read())
    L = []
    for (index, time, lose, draw, win) in res:
        lose = int(lose)
        draw = int(draw)
        win = int(win)
        tot = lose + draw + win
        episodes = int(index) * episodes_per_1
        L.append(Stat(episodes, float(time), win / tot, lose / tot, draw / tot))
    return L


DataFile = namedtuple("DataFile", ["name", "label", "episodes_per_1", "data"])


def fill_data(dfs):
    for df in dfs:
        df.data.extend(read_data("data/{}.txt".format(df.name), df.episodes_per_1))


def base_plot(df, part, cut=30, xlabel="Number of Episodes", jump=5):
    data = df.data[:cut]
    label = df.label
    with style.context("ggplot"):
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax1 = fig.add_subplot(1, 1, 1)
        ax2 = ax1.twiny()
        
        ax1.plot([i.episodes for i in data], [i.win_rate for i in data], "-o")
        
        ax1.set_ylim(0., 1.1)
        ax1.set_xlim(0, data[-1].episodes + data[0].episodes)
        ax2.set_xlim(0, data[-1].episodes + data[0].episodes)
        
        ax1.set_xticks([i.episodes for i in data[::jump]])
        ax1.set_xticklabels(["{:.1e}".format(i.episodes) for i in data[::jump]])
        ax2.set_xticks([i.episodes for i in data[::jump]])
        ax2.set_xticklabels(["{:.1e}".format(i.time) for i in data[::jump]])
        
        fig.suptitle("Win Rate of {}".format(label))
        fig.subplots_adjust(top=0.8)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("Win Rate")
        ax2.set_xlabel("Time (seconds)")
        fig.savefig("../docs/assets/figures/part{}/plot_{}.png".format(part, df.name))


def generate_plots(dfs, part):
    fill_data(dfs)
    for df in dfs:
        base_plot(df, part)


def compare_plots(dfs, part, cut=30, xlabel="Number of Episodes", jump=5,
    title="Comparison of Algorithms", show_time=False):
    with style.context("ggplot"):
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)
        
        for df in dfs:
            data = df.data[:cut] 
            ax.plot([i.episodes for i in data], [i.win_rate for i in data],
                "-o", label=df.label)
        
        ax.set_ylim(0., 1.1)
        ax.set_xlim(0, data[-1].episodes + data[0].episodes)
        ax.set_xticks([i.episodes for i in data[::jump]])
        ax.set_xticklabels(["{:.1e}".format(i.episodes) for i in data[::jump]])

        fig.suptitle(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Win Rate")

        if show_time:
            data = dfs[0].data
            ax2 = ax.twiny()
            ax2.set_xlim(0, data[-1].episodes + data[0].episodes)
            ax2.set_xticks([i.episodes for i in data[::jump]])
            ax2.set_xticklabels(["{:.1e}".format(i.time) for i in data[::jump]])
            ax2.set_xlabel("Time (seconds)")
            fig.subplots_adjust(top=0.8)
        
        ax.legend()
        
        fig.savefig("../docs/assets/figures/part{}/compare_{}.png".format(part,
            "_".join([df.name for df in dfs])))


def save_next_fig(part_num, fig):
    try:
        save_next_fig.fig_num += 1
    except AttributeError:
        save_next_fig.fig_num = 1
    fig.savefig("../docs/assets/figures/part{}/fig{}.png".format(part_num,
        save_next_fig.fig_num))
