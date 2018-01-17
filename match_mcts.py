from pathlib import Path
from argparse import ArgumentParser

from pong.utils.fast_evaluate_mcts import FastEvaluate
from pong.utils import common

    
parser = ArgumentParser()

parser.add_argument("save_path")
parser.add_argument("--n_episodes", "-ne", type=int, default=100)
parser.add_argument("--all_versions", action="store_true")
parser.add_argument("--self_play", action="store_true")

args = parser.parse_args()

FE = FastEvaluate(args.save_path, args.self_play)

if args.all_versions:
    p = Path(args.save_path)
    versions = [(int(v.parts[-1]), v) for v in p.iterdir() if v.is_dir()]
    versions.sort()
    t0 = versions[0][1].stat().st_mtime
    for i, v in versions:
        FE.base_policy.nn.load(str(v))
        score = FE.estimate(args.n_episodes)
        t1 = v.stat().st_mtime
        print("{i:04d}({t:0.2f}) : {l}|{draw}|{r}".format(i=i, t=t1-t0, **score))
else:
    score = FE.estimate(args.n_episodes)
    print("{l}|{draw}|{r}".format(**score))
