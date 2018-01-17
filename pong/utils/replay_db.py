import numpy as np

class ReplayDB:
    """Holds previous games and allows sampling random combinations of
        (state, action, new state, reward)
    """

    def __init__(self, state_dim, db_size):
        """Create new DB of size db_size."""

        self.state_dim = state_dim
        self.db_size = db_size
        self._empty_state = np.zeros((1, self.state_dim))

        self.DB = np.rec.recarray(self.db_size, dtype=[
            ("s1", np.float32, self.state_dim),
            ("s2", np.float32, self.state_dim),
            ("a", np.int32),
            ("r", np.float32),
            ("done", np.bool)
        ])
        self.clear()

    def clear(self):
        """Remove all entries from the DB."""

        self.index = 0
        self.n_items = 0
        self.full = False

    def store(self, s1, s2, a, r, done):
        """Store new samples in the DB."""

        n = s1.shape[0]
        if self.index + n > self.db_size:
            self.full = True
            l = self.db_size - self.index
            if l > 0:
                self.store(s1[:l], s2[:l], a[:l], r[:l], done[:l])
            self.index = 0
            if l < n:
                self.store(s1[l:], s2[l:], a[l:], r[l:], done[l:])
        else:
            v = self.DB[self.index : self.index + n]
            v.s1 = s1
            v.s2 = s2
            v.a = a
            v.r = r
            v.done = done
            self.index += n

        self.n_items = min(self.n_items + n, self.db_size)

    def sample(self, sample_size=None):
        """Get a random sample from the DB."""

        if self.full:
            db = self.DB
        else:
            db = self.DB[:self.index]

        if (sample_size is None) or (sample_size > self.n_items):
            return db
        else:
            return np.rec.array(np.random.choice(db, sample_size, False))

    def iter_samples(self, sample_size, n_samples):
        """Iterate over random samples from the DB."""
        
        if sample_size == 0:
            sample_size = self.n_items

        ind = self.n_items
        for i in range(n_samples):
            end = ind + sample_size
            if end > self.n_items:
                ind = 0
                end = sample_size
                p = np.random.permutation(self.n_items)
                db = np.rec.array(self.DB[p])
            yield db[ind : end]
            ind = end

    def store_episodes_results(self, results):
        """Store all results from episodes (in the format of
        greenlet_learner.)"""

        for r in results:
            done = np.zeros(r.states.shape[1], np.bool)
            done[-1] = True
            for i in range(r.states.shape[0]):
                s2 = np.vstack([r.states[i, 1:], self._empty_state])
                self.store(r.states[i], s2, r.actions[i], r.rewards[i], done)
