from argparse import ArgumentParser

from matplotlib import pyplot as plt

from pong.mechanics import pong, gui
from pong.utils import common


choices = ["manual", "random", "follow", "predict", "dpredict", "planning",
    "nn", "targetnn", "mcts"]

parser = ArgumentParser()
common.add_policy_args(parser, "l", choices, "follow")
common.add_policy_args(parser, "r", choices, "manual")

parser.add_argument("--disc", "-d", action="store_true")
parser.add_argument("--num_episodes", "-ne", type=int, default=float("inf"))
parser.add_argument("--capture", "-c")
parser.add_argument("--name_left", "-nl")
parser.add_argument("--name_right", "-nr")

args = parser.parse_args()

l_pol = common.read_policy_args(args, "l")
r_pol = common.read_policy_args(args, "r")

if args.disc:
    assert(args.right == "planning")
    sim = pong.Pong(f=r_pol.discretization)
else:
    sim = pong.Pong()

window = gui.GUI(sim, l_pol, r_pol, args.num_episodes,
    args.name_left, args.name_right, args.capture)
window.start()
plt.show()
window.end()
