---
layout: post
title: ApproxiPong
comments: true
image: "/assets/figures/part0/fig4.png"
---

Reinforcement Learning (RL) is all the rage nowadays. It doesn’t matter if you want to play [Atari 2600 games][atari], to master [go][alpha-go] ([twice][alpha-go-zero]) or even [chess][alpha-zero], you should probably use RL. Almost two years ago [Andrej Karpathy][karpathy] wrote an excellent post [“Pong from Pixels”][pong-pixels]. He did a wonderful job at explaining the RL problem and a specific RL algorithm (Policy Gradient) by the example of learning to play Pong. Here we extend his work, and compare multiple approaches to learn Pong. In contrast to the work of Kaparthy or DeepMind, we do not solve Pong “from pixels”, but from the actual internal state of the game. It’s easier, and allows us to focus on the RL part of the problem rather than the image processing part.

We rely on deep learning as a building block within classical RL algorithms. Solving Pong is not a very challenging task. The focus of this project is on understanding the merits and pitfalls of RL algorithms when combined with deep learning. 
 
 The structure of this website is as follows:
 
- In the [Introduction][intro] we give a very short introduction to RL, to make sure we’re all on the same page, and expand on what are we trying to do here.

- Perhaps the simplest approach to solve RL problems is by imitating an expert. In [Chapter 1][chapter1] we describe this approach and underscore its limitations.

- In [Chapter 2: When You Know the Model][chapter2] we present the most common mathematical formulation of the problem (MDP), and some classic solutions. We also show how they actually work when applied to Pong.

- In [Chapter 3: AlphaZero][chapter3] we present DeepMind’s breakthrough algorithm AlphaGo Zero, and our own implementation, AlphaPong Zero.

- In [Chapter 4: Learning While Playing Part 1][chapter4] we move on to learn how to play Pong without knowing the rules in advance, and present the Policy Gradient algorithm. We also give a variant of this algorithm with improved peformance.

- In [Chapter 5: Learning While Playing Part 2][chapter5] we present DeepMind’s Deep-Q-Learning algorithm, and some previously published variants and some of our own making.

- In [Chapter 6: Learning While Playing Part 3][chapter6] we present Actor-Critic methods: methods that attempt to combine the other algorithms we described. This combination does indeed obtain the best results in our experiments.

- Finally, there are some [References][references].

Should you read this article? If you’re interested in RL, but can’t find your feet in all the mess of concepts and algorithms, you should certainly read this. If you’re an RL expert, you probably already know most of it, but you might find some of the empirical results interesting. If you’re not at all interested in RL, then you should most certainly not read this article. Here, play some Pong instead:

{% include pong.html %}

Anyway, we hope you’ll enjoy reading through. And if you don’t know yet where to begin, simply start at the [Introduction][intro].

[atari]: https://deepmind.com/research/dqn/
[alpha-go]: https://deepmind.com/research/alphago/
[alpha-go-zero]: https://deepmind.com/blog/alphago-zero-learning-scratch/
[alpha-zero]: https://arxiv.org/abs/1712.01815
[karpathy]: http://karpathy.github.io/
[pong-pixels]: http://karpathy.github.io/2016/05/31/rl/
[intro]: {{ site.baseurl }}{% post_url 2018-01-11-introduction %}
[chapter1]: {{ site.baseurl }}{% post_url 2018-01-12-chapter1 %}
[chapter2]: {{ site.baseurl }}{% post_url 2018-01-13-chapter2 %}
[chapter3]: {{ site.baseurl }}{% post_url 2018-01-14-chapter3 %}
[chapter4]: {{ site.baseurl }}{% post_url 2018-01-15-chapter4 %}
[chapter5]: {{ site.baseurl }}{% post_url 2018-01-16-chapter5 %}
[chapter6]: {{ site.baseurl }}{% post_url 2018-01-17-chapter6 %}
[references]: {{ site.baseurl }}{% post_url 2018-01-18-references %}
