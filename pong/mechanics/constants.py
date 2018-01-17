#===============================================================================
# Game Parameters
#===============================================================================

TOP = 1.
BOTTOM = -1.
LEFT = -1.
RIGHT = 1.
HPL = 0.2

FRICTION = 0.6

A_STAY = 0
A_DOWN = 1
A_UP = 2

ACTIONS = {A_STAY, A_DOWN, A_UP}

DY = {
    A_UP: 0.05,
    A_STAY: 0.,
    A_DOWN: -0.05
}

VX = 0.03
VY0 = 0.01
VY1 = 0.05

MAX_STEPS = 1000


#===============================================================================
# GUI
#===============================================================================

# FIGSIZE = (3, 3)
FIGSIZE = (5, 5)
CLR_WHITE = (1,1,1)
CLR_GRAY = (100.0/255,100.0/255,100.0/255)
CLR_GREEN = (0,1,0)
CLR_BLACK = (0,0,0)
BALL_RADIUS = 0.03
BALL_COLOR = "g"
PADDLE_WIDTH = 0.03
PADDLE_COLOR = "r"

NAME_COLOR = "lightblue"
SCORE_COLOR = "y"

ARROW_START = 0.25
ARROW_LENGTH = 0.5
ARROW_COLOR = "y"
ARROW_WIDTH = 0.05

POINT_DELAY = 1.
FRAME_DELAY = 0.01
CAPTURE_FPS = 50
