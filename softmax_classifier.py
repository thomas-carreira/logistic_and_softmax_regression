import numpy as np
from random import shuffle
from classifier import Classifier


class Softmax(Classifier):
    """A subclass of Classifier that uses the Softmax to classify."""
    def __init__(self, random_seed=0):
        super().__init__('softmax')
        if random_seed:
            np.random.seed(random_seed)

    def loss(self, X, y=None, reg=0):
        scores = None
        # Initialize the loss and gradient to zero.
        loss = 0.0
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
        # TODO: Compute the softmax loss and store the loss in loss.                #
        # If you are not careful here, it is easy to run into numeric instability.  #
        # Don't forget the regularization!                                          #
        #############################################################################
        # number of classes
        c = y.max()+1
        y_one_hot = np.zeros((y.size, c))
        y_one_hot[np.arange(y.size), y] = 1
        q = np.exp(scores)/np.tile(np.array([np.sum(np.exp(scores), axis=1)]).T, (1, c))
        q = q + np.sum(reg * self.W.T.dot(self.W), axis=1)
        loss = -np.sum(y_one_hot * np.log(q))/q.shape[0]
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        
        # grad
        #############################################################################
        # TODO: Compute the gradients and store the gradients in dW.                #
        # Don't forget the regularization!                                          #
        #############################################################################
        dW = X.T.dot(q - y_one_hot) + reg*2*self.W
        dW = dW/len(y_one_hot)
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, dW

    def predict(self, X):
        y_pred = np.zeros(X.shape[1])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        scores = X.dot(self.W)
        c = self.W.shape[1]
        q = np.exp(scores)/np.tile(np.array([np.sum(np.exp(scores), axis=1)]).T, (1, c))
        y_pred = np.argmax(q, axis=1)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

