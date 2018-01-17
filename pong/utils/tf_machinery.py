from pathlib import Path

import numpy as np
import tensorflow as tf


class NeuralNetwork:

    @staticmethod
    def run_op_in_batches(session, op, batch_dict={}, batch_size=None,
        extra_dict={}):

        """Return the result of op by running the network on small batches of
        batch_dict."""

        if batch_size is None:
            return session.run(op, feed_dict={**batch_dict, **extra_dict})

        # Probably the least readable form to get an arbitrary item from a dict
        n = len(next(iter(batch_dict.values())))
        
        s = []
        for i in range(0, n, batch_size):
            bd = {k : b[i : i + batch_size] for (k, b) in batch_dict.items()}
            s.append(session.run(op, feed_dict={**bd, **extra_dict}))

        if s[0] is not None:
            if np.ndim(s[0]):
                return np.concatenate(s)
            else:
                return np.asarray(s)

    def __init__(self, input_dim, output_dim, hidden_layers, session=None,
        name_prefix="", input_=None):
        """Create an ANN with fully connected hidden layers of width
        hidden_layers.""" 
        
        self.saver = None
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name_prefix = name_prefix
        
        self.weights = []
        self.biases = []
        
        self.session = tf.Session() if session is None else session
        if input_ is None:
            self.input = tf.placeholder(tf.float32,
                shape=(None, self.input_dim),
                name="{}input".format(self.name_prefix)
            )
        else:
            self.input = input_
        self.layers = [self.input]

        for i, width in enumerate(hidden_layers):
            a = self.affine("{}hidden{}".format(self.name_prefix, i),
                self.layers[-1], width)
            self.layers.append(a)
        
        self.output = self.affine("{}output".format(self.name_prefix),
            self.layers[-1], self.output_dim, relu=False)
        self.probabilities = tf.nn.softmax(self.output,
            name="{}probabilities".format(self.name_prefix))
        self.output_max = tf.reduce_max(self.output, axis=1)
        self.output_argmax = tf.argmax(self.output, axis=1)
    
    def vars(self):
        """Iterate over all the variables of the network."""

        for w in self.weights:
            yield w
        for b in self.biases:
            yield b
    
    def affine(self, name_scope, input_tensor, out_channels, relu=True,
        residual=False):
        """Create a fully-connected affaine layer."""

        input_shape = input_tensor.get_shape().as_list()
        input_channels = input_shape[-1]
        with tf.variable_scope(name_scope):
            W = tf.get_variable("weights",
                initializer= tf.truncated_normal(
                    [input_channels, out_channels],
                    stddev=1.0 / np.sqrt(float(input_channels))
                ))
            b = tf.get_variable("biases",
                initializer=tf.zeros([out_channels]))
            
            self.weights.append(W)
            self.biases.append(b)
            
            A = tf.matmul(input_tensor, W) + b

            if relu:
                R = tf.nn.relu(A)
                if residual:
                    return R + input_tensor
                else:
                    return R
            else:
                return A

    def take(self, indices):
        """Return an operation that takes values from network outputs.
        e.g. NN.predict_max() == NN.take(NN.predict_argmax())
        """

        mask = tf.one_hot(indices=indices, depth=self.output_dim, dtype=tf.bool,
            on_value=True, off_value=False, axis=-1)
        return tf.boolean_mask(self.output, mask)
    
    def assign(self, other):
        """Return a list of operations that copies other network into self."""
        
        ops = []
        for (vh, v) in zip(self.vars(), other.vars()):
            ops.append(tf.assign(vh, v))
        return ops

    def reinit(self):
        """Reset weights to initial random values."""

        for w in self.weights:
            self.session.run(w.initializer)
        for b in self.biases:
            self.session.run(b.initializer)

    def save(self, save_path, step):
        """Save the current graph."""    
    
        if self.saver is None:
            with self.g.as_default():
                self.saver = tf.train.Saver(max_to_keep=None)

        p = Path(save_path)
        p.mkdir(parents=True, exist_ok=True)
        
        fname = str(p / "{:04d}".format(step) / "model.ckpt")
        self.saver.save(self.session, fname)

        return p / "{:04d}".format(step)

    def load(self, path):
        """Load weights or init variables if path==None."""

        if self.saver is None:
            self.saver = tf.train.Saver(max_to_keep=None)

        if path is None:
            self.session.run(tf.global_variables_initializer())
            return 0
        else:
            p = Path(path)

            files = p.glob("**/model.ckpt.meta")
            newest = max(files, key=lambda p: p.stat().st_ctime)
            fname = str(newest)[:-5]

            self.saver.restore(self.session, fname)
            
            return int(newest.parts[-2])

    def predict_probabilities(self, inputs_feed, batch_size=None):
        """Return softmax on NN outputs."""

        feed_dict = {self.input: inputs_feed}
        return self.run_op_in_batches(self.session, self.probabilities,
            feed_dict, batch_size)

    def predict_argmax(self, inputs_feed, batch_size=None):
        """Return argmax on NN outputs."""

        feed_dict = {self.input: inputs_feed}
        return self.run_op_in_batches(self.session, self.output_argmax,
            feed_dict, batch_size)

    def predict_max(self, inputs_feed, batch_size=None):
        """Return max on NN outputs."""
    
        feed_dict = {self.input: inputs_feed}
        return self.run_op_in_batches(self.session, self.output_max,
            feed_dict, batch_size)
    
    def predict_raw(self, inputs_feed, batch_size=None):
        """Return NN outputs without transformation."""

        feed_dict = {self.input: inputs_feed}
        return self.run_op_in_batches(self.session, self.output,
            feed_dict, batch_size)

    def predict_random(self, inputs_feed, epsilon=0.01, batch_size=None):
        """Return random element based on softmax on the NN outputs.
        epsilon is a smoothing parameter."""
    
        n = len(inputs_feed)
        base = self.predict_probabilities(inputs_feed, batch_size) + epsilon
        probs = base / base.sum(1, keepdims=True)
        out = np.zeros(n, np.int32)

        for i in range(n):
            out[i] = np.random.choice(self.output_dim, 1, p=probs[i])

        return out

    def predict_exploration(self, inputs_feed, epsilon=0.1, batch_size=None):
        """Return argmax with probability (1-epsilon), and random value with 
        probabilty epsilon."""

        n = len(inputs_feed)
        out = self.predict_argmax(inputs_feed, batch_size)
        exploration = np.random.random(n) < epsilon
        out[exploration] = np.random.choice(self.output_dim, exploration.sum())

        return out

    def train_in_batches(self, train_op, feed_dict, n_batches, batch_size,
        balanced=False):
        """Train the network by randomly sub-sampling feed_dict."""

        keys = tuple(feed_dict.keys())
        if balanced:
            ds = BalancedDataSet(*[feed_dict[k] for k in keys])
        else:
            ds = DataSet(*[feed_dict[k] for k in keys])

        for i in range(n_batches):
            batch = ds.next_batch(batch_size)
            d = {k : b for (k, b) in zip(keys, batch)}
            self.session.run(train_op, d)

    def accuracy(self, accuracy_op, feed_dict, batch_size):
        """Return the average value of an accuracy op by running the network
        on small batches of feed_dict."""

        return self.run_op_in_batches(self.session, accuracy_op,
            feed_dict, batch_size).mean()


class DataSet:
    """A class for datasets (labeled data). Supports random batches."""

    def __init__(self, *args):
        """Create a new dataset."""

        self.X = [a.copy() for a in args]
        self.n = self.X[0].shape[0]
        self.ind = 0
        self.p = np.random.permutation(self.n)

    def next_batch(self, batch_size):
        """Get the next batch of size batch_size."""

        if batch_size > self.n:
            batch_size = self.n

        if self.ind + batch_size > self.n:
            # we reached end of epoch, so we shuffle the data
            self.p = np.random.permutation(self.n)
            self.ind = 0
        
        batch = self.p[self.ind : self.ind + batch_size]
        self.ind += batch_size

        return tuple(a[batch] for a in self.X)


class BalancedDataSet:
    """A class for datasets (labeled data). Supports balanced random batches."""

    def __init__(self, X, l):
        """Create a new dataset."""

        labels = set(l)
        self.n_groups = len(labels)
        self.groups = []
        
        for label in labels:
            X_i = X[l == label]
            ds_i = DataSet(X_i, np.repeat(label, X_i.shape[0]))
            self.groups.append(ds_i)
        
        self.n = min(ds.n for ds in self.groups) * self.n_groups

    def next_batch(self, batch_size):
        """Get the next batch of size batch_size."""

        group_size = batch_size // self.n_groups
        
        X = []
        l = []
        
        for group in self.groups:
            X_i, l_i = group.next_batch(group_size)
            X.append(X_i)
            l.append(l_i)
        
        return np.vstack(X), np.hstack(l)
