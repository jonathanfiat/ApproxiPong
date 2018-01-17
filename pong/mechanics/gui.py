from time import time

from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.animation import FFMpegWriter

from .pong import Pong, S
from . import constants as c


class GUI:
    def __init__(self, sim, l_pol, r_pol, max_episodes,
        name_left=None, name_right=None, capture=None):
    
        self.sim = sim
        self.l_pol = l_pol
        self.r_pol = r_pol
        self.max_episodes = max_episodes
        self.l_name = name_left if name_left else self.l_pol.name
        self.r_name = name_right if name_right else self.r_pol.name
        self.capture = capture
        
        self.next_update = 0.
        self.episode_num = 0
        
        self.fig = plt.figure(figsize=c.FIGSIZE)
        self.canvas = self.fig.canvas
        self.canvas.set_window_title('PONG')

        self.ax = self.fig.add_axes((0., 0., 1., 1.))
        self.ax.axis([
            c.LEFT - c.PADDLE_WIDTH - c.BALL_RADIUS,
            c.RIGHT + c.PADDLE_WIDTH + c.BALL_RADIUS,
            c.BOTTOM - c.BALL_RADIUS,
            c.TOP + c.BALL_RADIUS,
        ])
        self.ax.tick_params(top="off", bottom="off", left="off", right="off")
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_aspect(1)
        self.ax.set_facecolor(c.CLR_BLACK)

        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        
        self.l = patches.Rectangle((0., 0.), c.PADDLE_WIDTH, 2 * c.HPL,
            color=c.PADDLE_COLOR, animated=True)
        self.r = patches.Rectangle((0., 0.), c.PADDLE_WIDTH, 2 * c.HPL,
            color=c.PADDLE_COLOR, animated=True)
        self.ball = patches.Circle((0., 0.), radius=c.BALL_RADIUS,
            color=c.BALL_COLOR, animated=True)
        
        self.l_arrow = patches.FancyArrow(c.ARROW_START, 0., -c.ARROW_LENGTH,
            0., c.ARROW_WIDTH, color=c.ARROW_COLOR, animated=True)
        self.r_arrow = patches.FancyArrow(-c.ARROW_START, 0., c.ARROW_LENGTH, 
            0., c.ARROW_WIDTH, color=c.ARROW_COLOR, animated=True)
        self.d_arrow = patches.FancyArrow(0., c.ARROW_START, 0.,
            -c.ARROW_LENGTH, c.ARROW_WIDTH, color=c.ARROW_COLOR, animated=True)
        
        self.ax.add_patch(self.l)
        self.ax.add_patch(self.r)
        self.ax.add_patch(self.ball)

        self.l_action = 0
        self.r_action = 0
        self.buttons = set()

        font_dict = {"family": "monospace", "size": "large", "weight": "bold",
            "animated": True}

        self.l_text = self.ax.text(c.LEFT, c.BOTTOM, self.l_name,
            color=c.NAME_COLOR, ha="left", **font_dict)
        self.r_text = self.ax.text(c.RIGHT, c.BOTTOM, self.r_name,
            color=c.NAME_COLOR, ha="right", **font_dict)
        self.score = self.ax.text((c.LEFT + c.RIGHT) / 2, c.BOTTOM, "",
            color=c.SCORE_COLOR, ha="center", **font_dict)

    def draw(self):
        
        if self.l_arrow.is_figure_set(): self.l_arrow.remove()
        if self.r_arrow.is_figure_set(): self.r_arrow.remove()
        if self.d_arrow.is_figure_set(): self.d_arrow.remove()
        
        self.score.set_text("{l}|{draw}|{r}".format(**self.sim.score))
        
        s = self.sim.get_state()
        
        self.ball.center = (s[S.BALL_X], s[S.BALL_Y])
        self.l.set_xy((c.LEFT - c.PADDLE_WIDTH - c.BALL_RADIUS,
            s[S.L_Y] - c.HPL))
        self.r.set_xy((c.RIGHT + c.BALL_RADIUS, s[S.R_Y] - c.HPL))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.l_text)
        self.ax.draw_artist(self.r_text)
        self.ax.draw_artist(self.score)

        if self.sim.done:
            if self.sim.win == "l":
                self.ax.add_patch(self.l_arrow)
                self.ax.draw_artist(self.l_arrow)
            elif self.sim.win == "r":
                self.ax.add_patch(self.r_arrow)
                self.ax.draw_artist(self.r_arrow)
            else:
                self.ax.add_patch(self.d_arrow)
                self.ax.draw_artist(self.d_arrow)
        else:
            self.ax.draw_artist(self.ball)
        
        self.ax.draw_artist(self.l)
        self.ax.draw_artist(self.r)
        self.canvas.blit(self.ax.bbox)
        if self.capture:
            if self.sim.done:
                n = c.CAPTURE_FPS // 2
            else:
                n = 1
            for i in range(n):
                self.writer.grab_frame()

    def main_loop(self):
        if (time() > self.next_update) or self.capture:
            state = self.sim.get_state()
            l_a = self.l_pol.get_action(state, self.buttons)
            r_a = self.r_pol.get_action(state, self.buttons)

            self.sim.step(l_a, r_a)
            self.draw()
            self.last_update = time()
            
            if self.sim.done:
                self.episode_num += 1
                if self.episode_num >= self.max_episodes:
                    plt.close()
                else:
                    self.sim.new_episode()
                    self.l_pol.new_episode()
                    self.r_pol.new_episode()
                    self.next_update = time() + c.POINT_DELAY
            else:
                self.next_update = time() + c.FRAME_DELAY

    def key_press(self, event):
        if event.key == "q":
            plt.close()
        else:
            self.buttons.add(event.key)

    def key_release(self, event):
        self.buttons.remove(event.key)

    def handle_redraw(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def first_draw(self, event):
        if self.canvas.manager.key_press_handler_id is not None:
            self.canvas.mpl_disconnect(self.canvas.manager.key_press_handler_id)

        self.canvas.mpl_disconnect(self.cid)
        self.canvas.mpl_connect('draw_event', self.handle_redraw)
        self.canvas.mpl_connect("key_press_event", self.key_press)
        self.canvas.mpl_connect("key_release_event", self.key_release)
        self.canvas.restore_region(self.background)

        self.timer = self.canvas.new_timer(interval=1)
        self.timer.add_callback(self.main_loop)
        self.timer.start()

    def start(self):
        self.cid = self.canvas.mpl_connect('draw_event', self.first_draw)
        if self.capture:
            self.writer = FFMpegWriter(fps=c.CAPTURE_FPS)
            self.writer.setup(self.fig, self.capture)
    
    def end(self):
        if self.capture:
            self.writer.finish()
