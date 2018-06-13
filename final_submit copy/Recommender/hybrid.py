import numpy as np
from surprise import KNNWithMeans, AlgoBase
from sklearn.linear_model import LinearRegression
from surprise import Dataset, Reader, Prediction


class WeightedHybrid(AlgoBase):

    _components = None
    _weights = None
    trainset = None
    

    def __init__(self, components):
        """ Constructor for WeightedHybrid

        :param components: The list of components to include in the hybrid
        :param normalize: If True, then weights are normalized to sum to 1 when set.
        """
        AlgoBase.__init__(self)

        # Set instance variables
        self._components = components
        
        


    def set_weights(self, weights):
        """ Set the hybrid weights and normalize.

        :param weights: New weights
        :return: No value
        """
        self._weights = weights
        self.normalize_weights()
        

    def normalize_weights(self):
        """ Normalize weight vector.

        Negative weights set to zero, and whole vector sums to 1.0.

        :return: No value
        """

        # Set negative weights to zero
        weights = np.array(self.get_weights())
        #np.where(weights > 0, weights, 0)
        weights[weights < 0] = 0
        weights = weights.ravel()
        
        
        # Normalize to sum to one.
        # If the weights are all zeros, set weights equal to 1/k, where k is the number
        # of components.
        all_weights = weights.sum()
        if all_weights == 0:
            self._weights = np.ones(len(weights)) * 1 / len(weights)
        else:
            self._weights = weights / all_weights
        

    def get_weights(self):
        return self._weights

    def fit(self, trainset):
        """ Fitting procedure for weighted hybrid.

        :param trainset:
        :return: No value
        """

        # Set the trainset instance variable
        self.trainset = trainset
        # Fit all of the components using the trainset.
        for i in self._components:
            i.fit(trainset)
            
        # Create arrays for call to LinearRegression function.
        # One array has dimensions [r, k] where r is the number of ratings and k is the number of components. The
        # array has the predictions for all the u,i pairs for all the components.
        # The other array has dimensions [r, 1] has all the ground truth rating values.
        rating_comp = np.zeros((self.trainset.n_ratings, len(self._components)))
        truth = []
        for i in range(len(self._components)):
            cur_comp = self._components[i]
            comp_rating = []            
            for uid, iid, rating in self.trainset.all_ratings():
                cur_rating = cur_comp.estimate(uid, iid)
                if isinstance(cur_rating, tuple):
                    cur_rating = cur_rating[0]
                #cur_rating = cur_comp.predict(str(uid), str(iid), r_ui = None, clip = False, verbose = False).est
                comp_rating.append(cur_rating)
                if len(truth) < self.trainset.n_ratings:
                    truth.append(rating)
            rating_comp[:,i] = np.array(comp_rating)
        truth_array = np.array(truth).reshape(-1,1)
        # Compute the LinearRegression.
        reg = LinearRegression(fit_intercept=True,copy_X = False)
        reg.fit(rating_comp, truth_array)
        # Set the weights.
        self.set_weights(reg.coef_)
        print("Learned weights {}".format(self.get_weights()))


    def predict(self, uid, iid, r_ui=None, clip=True, verbose=False):
        """ Predict a rating using the hybrid.

        :param uid: User id
        :param iid: Item id
        :param r_ui: Observed rating
        :param clip: If True, clip to rating range
        :param verbose:
        :return: A Prediction object
        """
       
        #est = self._intercept
        #Because we normalize the weights, so we should not use intercept as a part of prediction
        est = 0
        # Creat a new empty dictionary to retain any following Prediction "details" values 
        details={}
        weights = self.get_weights()
        
       
        for i in range(len(self._components)):
            comp = self._components[i]
            cur_pred = comp.predict(uid, iid, r_ui, clip = False)
            est += cur_pred.est * weights[i]
            details["Comp{0}".format(i)] = cur_pred.details
       
        if clip:
            low_bound, high_bound = self.trainset.rating_scale
            est = min(high_bound, est)
            est = max(low_bound, est)
                
        pred = Prediction(uid, iid, r_ui, est, details) 

        return pred
    
    def testme(self, uid, testset, verbose = False):
        predictions = []
        for i in range(len(testset)):
            if testset[i][0] == uid:
                pred = self.predict(uid, testset[i][1], testset[i][2])
                predictions.append(pred)
        return predictions
