from collections import namedtuple

import numpy as np
from scipy import interpolate
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import patches, lines, style
from mpl_toolkits.mplot3d import axes3d
from matplotlib.animation import FFMpegWriter

import common

PART_NUM = 1


def generate_fig1(width=5):
    fig = Figure(figsize=(width, 5))
    canvas = FigureCanvas(fig)
    ax = fig.add_axes((0., 0., 1., 1.))
    common.set_ax_params(ax)
    ax.set_facecolor((0., 0., 0.))
    ax.axis([
        common.LEFT - common.PADDLE_WIDTH - common.BALL_RADIUS,
        common.RIGHT + common.PADDLE_WIDTH + common.BALL_RADIUS,
        common.BOTTOM - common.BALL_RADIUS,
        common.TOP + common.BALL_RADIUS,
    ])

    l = patches.Rectangle(
        (common.LEFT - common.PADDLE_WIDTH - common.BALL_RADIUS, 0.6 - common.HPL),
        common.PADDLE_WIDTH, 2 * common.HPL, color=common.PADDLE_COLOR)
    r = patches.Rectangle(
        (common.RIGHT + common.BALL_RADIUS, - 0.5 - common.HPL),
        common.PADDLE_WIDTH, 2 * common.HPL, color=common.PADDLE_COLOR)
    ball = patches.Circle((0.6, 0.6), radius=common.BALL_RADIUS, color=common.BALL_COLOR)

    a = patches.FancyArrow(
        0.6, 0.6, 0.2, 0.06,
        width=0.01,
        color="pink")

    ax.add_patch(a)
    ax.add_patch(l)
    ax.add_patch(r)
    ax.add_patch(ball)

    font_dict = {"family": "monospace", "size": "large", "weight": "bold"}
    l_text = ax.text(common.LEFT, common.BOTTOM, "Follow",
        color=common.NAME_COLOR, ha="left", **font_dict)
    r_text = ax.text(common.RIGHT, common.BOTTOM, "Not Predict",
        color=common.NAME_COLOR, ha="right", **font_dict)
    
    common.save_next_fig(PART_NUM, fig)


def generate_fig2():
    fig = Figure(figsize=(8, 4))
    canvas = FigureCanvas(fig)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    for ax in [ax1, ax2]:
        common.set_ax_params(ax)
        ax.axis([0., 1., 0., 1.])
        ax.add_patch(patches.Ellipse((0.65, 0.3), 0.6, 0.3, 20, alpha=0.5,
            edgecolor="None", facecolor="pink"))
        ax.add_patch(patches.Ellipse((0.65, 0.3), 0.6, 0.3, 20,
            edgecolor="pink", facecolor="None", linewidth=3))
    
    e = patches.Ellipse((0.65, 0.3), 0.6, 0.3, 20, alpha=0.5, edgecolor="None",
        facecolor="pink", label='Possible "Predict" States')
    
    x = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5]
    y = [0.2, 0.25, 0.2, 0.25, 0.22, 0.36, 0.4, 0.37, 0.4, 0.32]
    tck, u = interpolate.splprep([x, y], s = 0.002, k=5)
    out = interpolate.splev(np.linspace(0, 1, 1000), tck)
    s = lines.Line2D(out[0], out[1], color="blue", label='Example "Predict" Trajectory')
    win = patches.Circle((x[-1], y[-1]), radius=0.01, color="blue", label='Episode End')
    ax1.add_artist(s)
    ax1.add_patch(win)
    ax1.legend(handles=[e, s, win],
        edgecolor="black",
        loc="upper left",
        fontsize="x-small")
    ax1.set_title("Real Expert")
    
    noise_x = (out[0] + np.random.normal(0, 0.006, 1000))[:960:10]
    noise_y = (out[1] + np.random.normal(0, 0.006, 1000))[:960:10]
    rand_x = np.linspace(noise_x[-1], 0.4, 10)[1:] + np.random.normal(0, 0.015, 9)
    rand_y = np.linspace(noise_y[-1], 0.6, 10)[1:] + np.random.normal(0, 0.015, 9)
    okay = lines.Line2D(np.r_[noise_x, rand_x[:1]], np.r_[noise_y, rand_y[:1]],
        color="blue",
        label='"Imitation" Trajectory inside D')
    rand = lines.Line2D(rand_x[0:], rand_y[0:], color="red", 
        label='"Imitation" Trajectory outside D')
    lose = patches.Circle((rand_x[-1], rand_y[-1]), radius=0.01, color="red",
        label="Episode End")
    ax2.add_artist(okay)
    ax2.add_artist(rand)
    ax2.add_patch(lose)
    ax2.legend(handles=[e, okay, rand, lose],
        edgecolor="black",
        loc="upper left",
        fontsize="x-small")
    ax2.set_title("Imitation")
    
    common.save_next_fig(PART_NUM, fig)



def generate_fig3():
    fig = Figure(figsize=(8, 8))
    canvas = FigureCanvas(fig)
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    
    E = []
    for ax in [ax1, ax2, ax3, ax4]:
        common.set_ax_params(ax)
        ax.axis([0., 1., 0., 1.])
        e1 = patches.Ellipse((0.65, 0.3), 0.6, 0.3, 20, alpha=0.5,
            edgecolor="None", facecolor="pink")
        e2 = patches.Ellipse((0.65, 0.3), 0.6, 0.3, 20, edgecolor="pink",
            facecolor="None", linewidth=3)
        ax.add_patch(e1)
        ax.add_patch(e2)
        E.append(e2)
    
    e0 = patches.Ellipse((0.65, 0.3), 0.6, 0.3, 20, alpha=0.5, edgecolor="None",
        facecolor="pink", label='Possible "Predict" States')

    x = np.random.random(500)
    y = np.random.random(500)
    s = lines.Line2D(x, y, linestyle="None", marker=".", markersize=2,
        label="Samples", color="darkblue")
    ax1.add_artist(s)
    s.set_clip_path(E[0])
    ax1.legend(handles=[e0, s],
        edgecolor="black",
        loc="upper left",
        fontsize="x-small")
    ax1.set_title("Sampling from Games")
    
    x = np.random.random(100)
    y = np.random.random(100)
    s = lines.Line2D(x, y, linestyle="None", marker=".", markersize=2, 
        label="Samples", color="darkblue")
    ax2.add_artist(s)
    ax2.legend(handles=[e0, s],
        edgecolor="black",
        loc="upper left",
        fontsize="x-small")
    ax2.set_title("Sampling Uniformally")
    
    e1 = patches.Ellipse((0.65, 0.3), 0.65, 0.35, 20, alpha=0.1,
        edgecolor="None", facecolor="blue", label="Useful Approximation")
    ax3.add_patch(e1)
    ax3.legend(handles=[e0, e1],
        edgecolor="black",
        loc="upper left",
        fontsize="x-small")
    
    e2 = patches.Rectangle((0., 0.), 1., 1., alpha=0.1, edgecolor="None",
        facecolor="blue", label="Useful Approximation")
    ax4.add_patch(e2)
    ax4.legend(handles=[e0, e2],
        edgecolor="black",
        loc="upper left",
        fontsize="x-small")
    
    common.save_next_fig(PART_NUM, fig)


Stat = namedtuple("Stat", ["epoch", "g_accuracy", "u_accuracy", "win_rate"])

def read_data(basedir="data/imitation data/v/", n=4):
    L = []
    
    for i in range(4):
        G_acc = "{}/{}_accuracy.txt".format(basedir, i)
        U_acc = "{}/{}_accuracy_artificial.txt".format(basedir, i)
        mat = "{}/{}_match.txt".format(basedir, i)
        for l1, l2, l3 in zip(open(G_acc), open(U_acc), open(mat)):
            a, b = l1.split(" : ")
            c, d = l2.split(" : ")
            e, f = l3.split(" : ")
            l, d, w = f.split("|")

            epoch = (300000. / 2000000.) * int(a)
            g_accuracy = float(b)
            u_accuracy = float(d)
            win_rate = int(w) / 1000.
            L.append(Stat(epoch, g_accuracy, u_accuracy, win_rate))

    return L


def generate_fig4():
    
    with style.context("ggplot"):
    
        v_data = read_data("data/imitation data/v/")
        
        x = [i.g_accuracy for i in v_data]
        y = [i.win_rate for i in v_data]
        
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, y, ".")
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Win Rate")
        ax.set_title("Win Rate as a function of Accuracy\n(sample size=2000000, sampled from games)")
        common.save_next_fig(PART_NUM, fig)
        
        x = [i.epoch for i in v_data]
        y1 = [i.g_accuracy for i in v_data]
        y2 = [i.win_rate for i in v_data]

        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
        ax.plot(x, y1, ".", label="Accuracy")
        ax.plot(x, y2, ".", label="Win Rate")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.set_title("Win Rate and Accuracy as a function of Epoch\n(sample size=2000000, sampled from games)")
        common.save_next_fig(PART_NUM, fig)

        
        t_data = read_data("data/imitation data/t/")
        a_data = read_data("data/imitation data/a/")
        
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)
        
        x = [i.epoch for i in v_data]
        y = [i.win_rate for i in v_data]
        ax.plot(x, y, ".", label="Vanilla")
        
        x = [i.epoch for i in t_data]
        y = [i.win_rate for i in t_data]
        ax.plot(x, y, ".", label="Decomposed")
        
        x = [i.epoch for i in a_data]
        y = [i.win_rate for i in a_data]
        ax.plot(x, y, ".", label="Uniform Sampling")
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Win Rate")
        ax.legend()
        ax.set_title("Comparison of Different Settings\n(sample size=2000000)")
        common.save_next_fig(PART_NUM, fig)


SAMPLE_SIZES = [200000, 400000, 600000, 800000, 1000000, 1200000, 1400000,
    1600000, 1800000, 2000000]
DATA_FILES = [
    [common.DataFile("imitation_sample_size/{}/{}".format(j, i),
        "imitation {}".format(i), 600 * 500, [])
            for j in [0, 1, 2, 3]]
                for i in SAMPLE_SIZES
]


def generate_fig5():
    for dfs in DATA_FILES:
        common.fill_data(dfs)

    X = np.array(SAMPLE_SIZES)
    Y = np.array([i.episodes for i in DATA_FILES[0][0].data])
    Z_avg = np.zeros((X.shape[0], Y.shape[0]))
    Z_min = np.zeros((X.shape[0], Y.shape[0]))
    Z_max = np.zeros((X.shape[0], Y.shape[0]))

    for x, dfs in enumerate(DATA_FILES):
        for y in range(len(dfs[0].data)):
            z = [df.data[y].win_rate for df in dfs]
            Z_avg[x, y] = np.mean(z)
            Z_min[x, y] = min(z)
            Z_max[x, y] = max(z)

    with style.context("ggplot"):
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X[:, None], Y[None, :], Z_avg, rstride=3, cstride=8)
        # ax.plot_wireframe(X[:, None], Y[None, :], Z_min, rstride=3, cstride=8)
        # ax.plot_wireframe(X[:, None], Y[None, :], Z_max, rstride=3, cstride=8)
        ax.set_xlabel("Sample Size")
        ax.set_ylabel("SGD steps")
        ax.set_zlabel("Win Rate")
        ax.view_init(40, -60)
        ax.ticklabel_format(style="sci", scilimits=(-2, 2))

        common.save_next_fig(PART_NUM, fig)
        
        writer = FFMpegWriter(fps=20)
        writer.setup(fig, "figures/part{}/movie.mp4".format(PART_NUM))
        writer.grab_frame()
        
        for i in range(-60, 360 * 2 - 60, 1):
            ax.view_init(40, i)
            writer.grab_frame()
        writer.finish()


if __name__ == "__main__":
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    generate_fig1(10)
