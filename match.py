from pathlib import Path
from argparse import ArgumentParser

from pong.utils.fast_evaluate import FastEvaluate
from pong.utils import common

    
choices = ["manual", "follow", "predict", "planning", "nn", "targetnn"]

parser = ArgumentParser()
common.add_policy_args(parser, "l", choices, "follow")
common.add_policy_args(parser, "r", choices, "follow")

parser.add_argument("--disc", "-d", action="store_true")
parser.add_argument("--n_episodes", "-ne", type=int, default=100)

parser.add_argument("--all_versions", "-av", action="store_true")
parser.add_argument("--versions_jump", "-vj", type=int, default=1)

args = parser.parse_args()

if args.disc:
    assert(args.right == "planning")

l_pol = common.read_policy_args(args, "l")
r_pol = common.read_policy_args(args, "r")
FE = FastEvaluate(l_pol, r_pol, args.disc)

if args.all_versions:
    p = Path(r_pol.save_path)
    versions = [(int(v.parts[-1]), v) for v in p.iterdir() if v.is_dir()]
    versions.sort()
    t0 = versions[0][1].stat().st_mtime
    for i, v in versions[::args.versions_jump]:
        r_pol.nn.load(str(v))
        score = FE.estimate(args.n_episodes)
        t1 = v.stat().st_mtime
        print("{i:04d}({t:0.2f}) : {l}|{draw}|{r}".format(i=i, t=t1-t0, **score))
else:
    score = FE.estimate(args.n_episodes)
    print("{l}|{draw}|{r}".format(**score))
