---
layout: post
title:  "Conclusions & Questions"
date:   2018-01-18 12:00:00 +0200
comments: true
description: "Concluding remarks about the ApproxiPong project."
image: "/assets/figures/part0/fig2.png"
---

# Things that could appear here:

1. Real-life application of RL. What should be done so RL will become a technology. Which sould lead us to:
2. Beyond MDP: even if there is some underlying MDP, less than perfect observations ruins it (see what happened in the discretization case). So we would like to find a solution for non-MDP models.
3. Easy and hard exploration: what we didn’t cover at all (Montezuma's Revenge).
4. The Reduction of Multi-Agent problem (Go, Pong) to single agent vs. environment:
    1. Can make the learning easier (gradually increasing difficulty)
    2. Approximation of the MiniMax solution?
5. Proofs and deductive justifications: Q-learning, P-learning, Alpha-Go.

# References

## Books

1. [Algorithms for Reinforcement Learning][arl], by Csaba Szepesvári
2. [Reinforcement Learning: An Introduction][rlai], by Richard S. Sutton and Andrew G. Barto

## Articles

1. [Human-level control through deep reinforcement learning][hlctdrl]
2. [Mastering the game of Go with deep neural networks and tree search][mtgogwdnnats]
3. [Mastering the game of Go without human knowledge][mtgogwhk]
4. [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm][mcasbspwagrla]
5. [Deep Reinforcement Learning with Double Q-learning][drlwdql]

[arl]: https://sites.ualberta.ca/~szepesva/RLBook.html
[rlai]: http://incompleteideas.net/book/the-book-2nd.html
[hlctdrl]: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
[mtgogwdnnats]: https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf
[mtgogwhk]: http://www.nature.com/articles/nature24270.pdf
[mcasbspwagrla]: https://arxiv.org/abs/1712.01815
[drlwdql]: https://arxiv.org/abs/1509.06461
