# ApproxiPong

ApproxiPong is a small project designed to test different Reinfocement Learning algorithms on a specific task: Pong. Using many differet algorithms on this very simple task allows us to better understand their strengths and weaknesses. For a full description of th project, check our [website](https://jonathanfiat.github.io/ApproxiPong/).

You are more than welcome to read our code, play with it, change it to test new algorithms or adapt it for your own RL problems. However, be warned that this is not a RL library, and we didn't implement the different algorithms in a generic way. Applying them to anything other than Pong would require some work.

## Dependencies

In order to run our code, you'll need:

- [Python](https://www.python.org/)>=3.5
- [TensorFlow](https://www.tensorflow.org/)>=1.4
- [Numpy](http://www.numpy.org/)>=1.13
- [Greenlet](https://greenlet.readthedocs.io/)>=0.4.12

And if you want GUI and the ability to re-create our illustration, you'll also need:

- [Scipy](https://www.scipy.org/)>=1.0.0
- [Matplotlib](https://matplotlib.org/)>=2.1.0 (only for GUI and illustrations)

## Running the Code

### *learn.py*

Executing

    python learn.py algorithm

will run "algorithm". Every algorithm has many options of its own, but all of them support "--save_path" (where to save the results) and "--save_frequency" (how often to save). For example, running

    python learn.py imitation --train_size 2000000 --test_size 1000000 -num_iters 600 -sp /tmp/Pong/Imitation/

will run the "Imitation" algorithm.

### *play.py*

Executing

    python play.py

will open a window and will play Pong. By default, the left paddle is controlled by a simple policy called "Follow", and the right one is controlled by the user (using the arrows). Both paddles can be controlled by different policies:

    python play.py -r nn -ra /tmp/Pong/Imitation

will show games between "Follow" and the learnt policy stored in /tmp/Pong/Imitation.

### *match.py* and *match_mcts.py*

Executing

    python match.py

will perform many games between any two policies, in a much more efficient way than play.py. For example, 

    python match.py -r nn -ra /tmp/Pong/Imitation

will perform 100 games between "Follow" and the learnt policy stored in /tmp/Pong/Imitation.

*match_mcts.py* does the same thing, but specifically for the AlphaPongZero algorithm, and it will use MCTS while playing (making it much much slower - it can take a few hours to run 100 games).

# Directory Structure

## graphics/

This directory contains the code (and data) required to generate the illustrations in our website. Unless you want to create similar but not identical illustrations you can ignore it.

## pong/mechanics/

This directory contains all the code required for the Pong game - the game logic and the GUI. It also contains the file *policies.py* that implements the different policies.

## pong/learning/

This directory contains a single file for every learning algorithm, fully implementing that specific algorithm. If you want to understand or modify one of the learning algorithm, you should find the relevant file in this directory and start from there.

## pong/utils/

This directory contains everything that doesn't fit within the other two, mostly pieces of code that are common for more than one algorithm.

# Examples
