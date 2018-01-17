import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import patches, lines, gridspec, text
from matplotlib.animation import FFMpegWriter

import common

PART_NUM = 2


def generate_fig1():
    fig = Figure(figsize=(6, 4))
    canvas = FigureCanvas(fig)
    ax = fig.add_axes((0.01, 0.01, 0.98, 0.98))
    common.set_ax_params(ax)
    ax.axis([0., 1.5, 0., 1.])

    r = 0.05
    ax.add_patch(patches.FancyBboxPatch(
        (0.1 - r, 0.5 - r),
        width=2 * r,
        height=2 * r,
        boxstyle="round,pad=0.01",
        facecolor="lightblue"
    ))
    ax.text(0.1, 0.5, "$s$", ha="center", va="center", size="large")
    heights = np.linspace(0.8, 0.2, 3)
    x = np.linspace(0.3 + r + 0.01, 1.4, 10)

    for i in range(3):
        h = heights[i]
        
        for j in range(3):
            base = h + (j - 1) / 12.
            y = base + np.random.uniform(-1., 1., 10) / 30.
            y[0] = h + (j - 1) / 24.
            ax.add_artist(lines.Line2D(x, y, color="black"))
            ax.add_patch(patches.Circle((x[-1], y[-1]), 0.01, color="black"))
        
        ax.add_patch(patches.FancyBboxPatch(
            (0.3 - r, h - r),
            width=2 * r,
            height=2 * r,
            boxstyle="round,pad=0.01",
            facecolor="lightgreen"
        ))
        ax.text(0.3, h, "$a_{}$".format(i),
            ha="center", va="center", size="large")

        ax.add_patch(common.arrow_by_start_end(
            (0.1 + r + 0.01, 0.5 + r * (1 - i) / 3.), (0.3 - r - 0.01, h),
            length_includes_head=True, color="black", head_width=0.02))

    common.save_next_fig(PART_NUM, fig)


def generate_fig2():
    fig = Figure(figsize=(6, 4))
    canvas = FigureCanvas(fig)
    ax = fig.add_axes((0.01, 0.01, 0.98, 0.98))
    common.set_ax_params(ax)
    ax.axis([0., 1.5, 0., 1.])
    
    r = 0.05
    ax.add_patch(patches.FancyBboxPatch(
        (0.1 - r, 0.5 - r),
        width=2 * r,
        height=2 * r,
        boxstyle="round,pad=0.01",
        facecolor="lightblue"
    ))
    ax.text(0.1, 0.5, "$s$", ha="center", va="center", size="large")
    heights = np.linspace(0.8, 0.2, 3)
    x = np.linspace(0.5 + r/2 + 0.01, 1.4, 10)

    for i in range(3):
        h = heights[i]
        
        for j in range(3):
            h2 = h + (heights[j] - 0.5) / 3.5
            ax.add_patch(patches.FancyBboxPatch(
                (0.5 - r/2, h2 - r/2),
                width=r,
                height=r,
                boxstyle="round,pad=0.01",
                facecolor="lightgreen"
            ))
            ax.text(0.5, h2, "$a_{}$".format(j),
                ha="center", va="center", size="small")
            ax.add_patch(common.arrow_by_start_end(
                (0.3 + r + 0.01, h + r * (1 - j) / 3.),
                (0.5 - r/2 - 0.01, h2), length_includes_head=True,
                color="black", head_width=0.02))
            for k in range(2):
                base = h2 + (k - 0.5) / 24.
                y = base + np.random.uniform(-1., 1., 10) / 60.
                y[0] = h2 + (k - 0.5) / 24.
                ax.add_artist(lines.Line2D(x, y, color="black"))
                ax.add_patch(patches.Circle((x[-1], y[-1]), 0.01,
                    color="black"))
        
        ax.add_patch(patches.FancyBboxPatch(
            (0.3 - r, h - r),
            width=2 * r,
            height=2 * r,
            boxstyle="round,pad=0.01",
            facecolor="lightgreen"
        ))
        ax.text(0.3, h, "$a_{}$".format(i),
            ha="center", va="center", size="large")

        ax.add_patch(common.arrow_by_start_end(
            (0.1 + r + 0.01, 0.5 + r * (1 - i) / 3.),
            (0.3 - r - 0.01, h), length_includes_head=True, color="black",
            head_width=0.02))

    common.save_next_fig(PART_NUM, fig)


class RandomTree:
    def __init__(self, degree, parent=None):
        self.degree = degree
        self.children = [None] * degree
        self.active = False
        
        if parent is None:
            self.parent = None
            self.visitorder = [self]
        else:
            self.parent = parent
            self.visitorder = parent.visitorder
            self.visitorder.append(self)
        
        self.patches = {}
    
    def simulate(self, max_depth):
        if max_depth:
            a = np.random.choice(self.degree)
            if self.children[a] is None:
                self.children[a] = RandomTree(self.degree, self)
            self.children[a].simulate(max_depth - 1)
    
    def set(self):
        self.active = True
        for child in self.children:
            if child is not None:
                child.set()
    
    def draw(self, ax, facecolor, alpha=1., text_size="small"):
        L = []
        
        L.append(patches.FancyBboxPatch(
            self.box_xy,
            width=self.width,
            height=self.height,
            boxstyle="round,pad=0.1",
            facecolor=facecolor,
            alpha=alpha
        ))
        L.append(text.Text(self.xy[0], self.xy[1], self.text,
            ha="center", va="center", size=text_size, alpha=alpha))
        if self.parent:
            L.append(common.arrow_by_start_end(
                self.father_a_xy,
                self.a_xy,
                length_includes_head=True,
                color="black",
                head_width=0.1,
                alpha=alpha))
        
        for a in L:
            ax.add_artist(a)
        self.patches[ax] = L
    
    def remove(self, ax):
        for a in self.patches[ax]:
            a.remove()
        del self.patches[ax]


def generate_fig3():
    tree = RandomTree(3)
    for i in range(10):
        tree.simulate(8)
    
    while True:
        try:
            a = np.random.choice(3)
            tree.children[a].set()
            break
        except AttributeError:
            pass
    
    fig1 = Figure(figsize=(16/2, 9/2))
    canvas1 = FigureCanvas(fig1)
    ax1 = fig1.add_axes((0.01, 0.01, 0.98, 0.98))
    fig2 = Figure(figsize=(16/2, 9/2))
    canvas2 = FigureCanvas(fig2)
    ax2 = fig2.add_axes((0.01, 0.01, 0.98, 0.98))

    fig3 = Figure(figsize=(16, 9))
    canvas3 = FigureCanvas(fig3)
    ax3 = fig3.add_axes((0.01, 0.01, 0.98, 0.98))

    for ax in [ax1, ax2, ax3]:
        common.set_ax_params(ax)
        ax.axis([0., 16., 0., 9.])

    r = 0.4
    
    tree.xy = (1., 9. / 2.)
    tree.box_xy = (1. - r, 9. / 2. - r)
    tree.width = 2 * r
    tree.height = 2 * r
    tree.text = "$s$"
    tree.facecolor1 = "lightblue"
    tree.facecolor2 = "lightblue"
    tree.alpha = 0.2
    tree.connectors = [(1. + r + 0.1, 9. / 2. + j * r / 3) for j in [1, 0, -1]]

    X = np.linspace(3., 15., 8)
    L = [tree]
    for i in range(8):
        L2 = []
        for n in L:
            L2.extend(c for c in n.children if c)
        Y = np.linspace(9., 0., len(L2) + 2)[1:-1]
        cnt = 0
        for n in L:
            for j in range(3):
                if n.children[j] is not None:
                    c = n.children[j]
                    x, y = X[i], Y[cnt]
                    c.connectors = [(x + r/2 + 0.1, y + k * r / 6)
                        for k in [1, 0, -1]]
                    c.xy = (x, y)
                    c.box_xy = (x - r/2, y - r/2)
                    c.width = r
                    c.height = r
                    c.father_a_xy = n.connectors[j]
                    c.a_xy = (x - r/2 - 0.1, y)
                    c.text = "$a_{}$".format(j)
                    c.facecolor1 = "lightgreen"
                    if (i==0 and c.active):
                        c.facecolor2 = "lightblue"
                    else:
                        c.facecolor2 = "lightgreen"
                    c.alpha = 1. if c.active else 0.2
                    cnt += 1
        L = L2
    
    writer = FFMpegWriter()
    writer.setup(fig3, "figures/part{}/mcts_movie.mp4".format(PART_NUM))
    writer.grab_frame()
    reset = False
    
    for c in tree.visitorder:
        if reset:
            n = c
            L = []
            while n:
                L.append(n)
                n = n.parent
        else:
            L = [c]

        for n in L[::-1]:
            n.draw(ax3, "red", 1., "xx-large")
            writer.grab_frame()
            n.remove(ax3)
            n.draw(ax3, n.facecolor1, 1., "xx-large")
            writer.grab_frame()

        c.draw(ax1, c.facecolor1, 1.)
        c.draw(ax2, c.facecolor2, c.alpha)
        
        reset = not any(c.children)
    
    writer.finish()
    
    common.save_next_fig(PART_NUM, fig1)
    common.save_next_fig(PART_NUM, fig2)


def generate_fig4():
    fig = Figure(figsize=(8, 8))
    canvas = FigureCanvas(fig)
    fig.suptitle("Demonstration of How Descritezation Creates Non-Determinism")
    
    gs = gridspec.GridSpec(2, 3)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_aspect(1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_facecolor((0., 0., 0.))
        ax.tick_params("both", bottom="off", left="off")
        ax.axis([-1., 1., -1., 1.])
        x_axis = lines.Line2D([-1., 1.], [0., 0.], color="red", alpha=0.5,
            linestyle="--")
        y_axis = lines.Line2D([0., 0.], [-1., 1.], color="red", alpha=0.5,
            linestyle="--")
        ax.add_artist(x_axis)
        ax.add_artist(y_axis)
    
    ax1.set_title("Current Partial State")
    c_pos = patches.Rectangle((-1., -1.), 1., 1., color="palevioletred",
        alpha=0.7)
    a = patches.FancyArrow(-0.5, -0.5, 0.4, 0.4, width=0.01, color="pink")
    ax1.add_patch(c_pos)
    ax1.add_patch(a)
    
    ax2.set_title("Possible Full State 1")
    c_pos = patches.Rectangle((-1., -1.), 1., 1., color="palevioletred",
        alpha=0.7)
    ball = patches.Circle((-0.8, -0.2), radius=2 * common.BALL_RADIUS,
        color=common.BALL_COLOR)
    a = patches.FancyArrow(-0.8, -0.2, 0.4, 0.4, width=0.01, color="pink")
    ax2.add_patch(c_pos)
    ax2.add_patch(a)
    ax2.add_patch(ball)
    
    ax3.set_title("Possible Full State 2")
    c_pos = patches.Rectangle((-1., -1.), 1., 1., color="palevioletred",
        alpha=0.7)
    ball = patches.Circle((-0.2, -0.2), radius=2 * common.BALL_RADIUS,
        color=common.BALL_COLOR)
    a = patches.FancyArrow(-0.2, -0.2, 0.4, 0.4, width=0.01, color="pink")
    ax3.add_patch(c_pos)
    ax3.add_patch(a)
    ax3.add_patch(ball)
    
    ax4.set_title("Possible Full State 3")
    c_pos = patches.Rectangle((-1., -1.), 1., 1., color="palevioletred",
        alpha=0.7)
    ball = patches.Circle((-0.2, -0.8), radius=2 * common.BALL_RADIUS,
        color=common.BALL_COLOR)
    a = patches.FancyArrow(-0.2, -0.8, 0.4, 0.4, width=0.01, color="pink")
    ax4.add_patch(c_pos)
    ax4.add_patch(a)
    ax4.add_patch(ball)
    
    common.save_next_fig(PART_NUM, fig)


def generate_fig5():    
    def possible_v(ax, dir1, dir2, size, start):
        t1 = np.arctan2(*dir1)
        t2 = np.arctan2(*dir2)
        a1 = common.arrow_by_start_size_angle(start, size, t1, width=0.01,
            color="pink")
        a2 = common.arrow_by_start_size_angle(start, size, t2, width=0.01,
            color="pink")
        arc = patches.Arc(start, size, size, 0.0, np.degrees(t1),
            np.degrees(t2), color="pink")
        ax.add_patch(a1)
        ax.add_patch(a2)
        ax.add_patch(arc)
    
    fig = Figure(figsize=(8, 8))
    canvas = FigureCanvas(fig)
    fig.suptitle("Demonstration of How Descritezation Creates Non-Markovian Models")
    
    gs = gridspec.GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    for ax in [ax1, ax2, ax3]:
        common.set_ax_params(ax, "black")
        ax.axis([-1., 1., -1., 1.])
        for v in np.linspace(-1., 1., 5)[1:-1]:
            x_axis = lines.Line2D([-1., 1.], [v, v], color="red", alpha=0.5,
                linestyle="--")
            y_axis = lines.Line2D([v, v], [-1., 1.], color="red", alpha=0.5,
                linestyle="--")
            ax.add_artist(x_axis)
            ax.add_artist(y_axis)

    ax1.set_title("Current Partial State")
    c_pos = patches.Rectangle((0., 0.), 0.5, 0.5, color="palevioletred",
        alpha=0.7)
    ax1.add_patch(c_pos)
    possible_v(ax1, (0.2, 1.), (1., 0.2), 0.4, (0.25, 0.25))
    
    ax2.set_title("Possible Past 1 &\nImplications on Current State")
    c_pos = patches.Rectangle((0., 0.), 0.5, 0.5, color="palevioletred",
        alpha=0.7)
    p1_pos = patches.Rectangle((-0.5, 0.), 0.5, 0.5, color="palevioletred",
        alpha=0.5)
    p2_pos = patches.Rectangle((-1., 0.), 0.5, 0.5, color="palevioletred",
        alpha=0.3)
    ax2.add_patch(c_pos)
    ax2.add_patch(p1_pos)
    ax2.add_patch(p2_pos)
    possible_v(ax2, (0.2, 1.), (1., 1.), 0.4, (0.25, 0.25))
    
    ax3.set_title("Possible Past 2 &\nImplications on Current State")
    c_pos = patches.Rectangle((0., 0.), 0.5, 0.5, color="palevioletred",
        alpha=0.7)
    p1_pos = patches.Rectangle((0., -0.5), 0.5, 0.5, color="palevioletred",
        alpha=0.5)
    p2_pos = patches.Rectangle((0., -1.), 0.5, 0.5, color="palevioletred",
        alpha=0.3)
    ax3.add_patch(c_pos)
    ax3.add_patch(p1_pos)
    ax3.add_patch(p2_pos)
    possible_v(ax3, (1., 1.), (1., 0.2), 0.4, (0.25, 0.25))
    
    common.save_next_fig(PART_NUM, fig)


DATA_FILES = [
    common.DataFile("deep_value_iteration_follow", "Deep Q Iteration", 1, []),
    common.DataFile("deep_value_iteration_random2", "Deep Q Iteration Random2", 1, []),
    common.DataFile("deep_value_iteration_random", "Deep Q Iteration Random", 1, []),
    common.DataFile("deep_value_iteration_self", "Deep Q Iteration Self", 1, []),
]

if __name__ == "__main__":
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    common.fill_data(DATA_FILES)
    common.base_plot(DATA_FILES[0], PART_NUM, cut=50, xlabel="Q-Iterations",
        jump=10)
    # common.compare_plots(DATA_FILES, PART_NUM, xlabel="Q-Iterations", cut=100)
