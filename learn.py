from argparse import ArgumentParser
from importlib import import_module
import sys

ALGORITHMS = [
    "imitation",
    "value_iteration",
    "deep_value_iteration",
    "deep_q_deepmind",    
    "deep_q",
    "double_deep_q",
    "policy_gradient",
    "actor_critic",
    "success_learning",
    "success_learning_critic",
    "deep_p",
    "alpha_pong_zero",
]

parser = ArgumentParser()
parser.add_argument("algorithm", choices=ALGORITHMS)
args = parser.parse_args(sys.argv[1:2])

algorithm = import_module("pong.learning.{}".format(args.algorithm))
algorithm.main(sys.argv[2:])
