import numpy as np
from random import shuffle
from classifier import Classifier


class Logistic(Classifier):
    """A subclass of Classifier that uses the logistic function to classify."""
    def __init__(self, random_seed=0):
        super().__init__('logistic')
        if random_seed:
            np.random.seed(random_seed)

    def loss(self, X, y=None, reg=0):
        """
        Softmax loss function, vectorized version.

        Inputs and outputs are the same as softmax_loss_naive.
        """
        # Initialize the loss and gradient to zero.
        scores = None
        loss = None
        dW = np.zeros_like(self.W)
        num_classes = self.W.shape[1]
        num_train = X.shape[0]

        #scores
        #############################################################################
        # TODO: Compute the scores and store them in scores.                        #
        #############################################################################
        scores = X.dot(self.W)
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        if y is None:
            return scores


        # loss
        #############################################################################
        # TODO: Compute the logistic loss and store the loss in loss.               #
        # If you are not careful here, it is easy to run into numeric instability.  #
        # Don't forget the regularization!                                          #
        #############################################################################
        sigmoid = 1/(1 + np.exp(-scores))
        sigmoid = np.reshape(sigmoid, y.size)
        loss = (-y * np.log(sigmoid) - (1 - y) * np.log(1 - sigmoid)) + reg * self.W.T.dot(self.W)
        loss = loss.mean()
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        # grad
        #############################################################################
        # TODO: Compute the gradients and store the gradients in dW.                #
        # Don't forget the regularization!                                          #
        #############################################################################     
        dW = (sigmoid - y).dot(X) + (reg * 2 * np.reshape(self.W, len(self.W)))
        dW = dW/y.size
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        return loss, dW

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        scores = X.dot(self.W)
        sigmoid = 1 / (1 + np.exp(-scores))
        y_pred = np.where(sigmoid >= 0.5, 1, 0)
        y_pred = np.reshape(y_pred, len(X))
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################

        return y_pred
