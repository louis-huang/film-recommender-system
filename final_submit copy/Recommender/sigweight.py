import numpy as np
from surprise import KNNWithMeans
from six import iteritems

class KNNSigWeighting(KNNWithMeans):

    def __init__(self, k=40, min_k=1, sim_options=None, **kwargs):

        if sim_options is None:
            sim_options = {}

        KNNWithMeans.__init__(self, sim_options=sim_options, **kwargs)
        self.k = k
        self.min_k = min_k
        self.overlap = None
        
    def fit(self, trainset):
        """Model fitting for KNN with significance weighting

        Calls the parent class fit method and then generates the overlap matrix
        needed by the significance weighting.

        :param trainset:
        :return: self
        """

        # Call parent class function
        KNNWithMeans.fit(self, trainset)

        # Create an "overlap" matrix counting the number of items that
        # pairs of users have in common.
        # See the creation of the "freq" matrix in the "similarities.pyx" file.
        if self.sim_options['user_based']:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur

        self.overlap = np.zeros((n_x, n_x), np.int)
        for y, y_ratings in iteritems(yr):
            for xi, ri in y_ratings:
                for xj, rj in y_ratings:
                    self.overlap[xi, xj] += 1
        
        # Use overlap matrix to update the sim matrix, discounting by the significance weight factor.
        for xi in range(n_x):
            for xj in range(n_x):
                weight = self.sig_weight(xi, xj)
                self.sim[xi, xj] = self.sim[xi, xj] * weight
        return self

    def sig_weight(self, x1, x2):
        """Computes significance weight based on overlap and threshold.

        Threshold is provided in sim_options with key 'corate_threshold' with a default of 50.

        .. math::
           \\frac{
           min(|I_u \\cap I_v|, \\beta)}
           {\\beta}

        :param x1: user u
        :param x2: user v
        :return: the weight associated with the users x1 and x2
        """
        common = self.overlap[x1,x2]
        beta = self.sim_options.get('corate_threshold', 50)
        if common < beta:
            return common/beta
        
        return 1.0



