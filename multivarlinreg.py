import numpy as np

class LinearRegression:
    # A linear regression data structure. 
    # Heavily inspired of Jason Brownlee's implementation
    # Found at https://machinelearningmastery.com/implement-linear-regression-stochastic-gradient-descent-scratch-python/
    # It uses a random or stochastic gradient descent algorithm

    def __init__(self):
        self.xtrain_full = np.loadtxt('redwine_training.txt')
        self.xtrain      = self.xtrain_full[:, :11]
        self.ytrain      = self.xtrain_full[:, 11]

        self.xtest_full  = np.loadtxt('redwine_testing.txt')
        self.xtest       = self.xtest_full[:, :11]
        self.ytest       = self.xtest_full[:, 11]

        self.weights     = []
        print(self.xtrain.shape, self.xtest.shape)

    def _fs_validate(self):
        for row in self.xtrain:
            for entry in row:
                if entry < 0 or entry > 1:
                    print('train', row, entry)
                    return False
        return True

    def _predict(self, row, weights):
        yhat = weights[0]
        for i in range(len(row)):
            yhat += weights[i+1] * row[i]
        return yhat

    def _accuracy(self, predictions):
        sum_error = 0
        for i in range(len(predictions)):
            sum_error += abs(predictions[i] - self.ytest[i])
        return (sum(self.ytest) - sum_error) / sum(self.ytest)

    def feature_scaling(self):
        for c in np.arange(0, self.xtrain.shape[1]):
            c_min  = np.min(self.xtrain.T[c])
            c_max  = np.max(self.xtrain.T[c])
            self.xtrain.T[c] = (self.xtrain.T[c] - c_min) / (c_max - c_min)
            self.xtest.T[c]  = (self.xtest.T[c] - c_min) / (c_max - c_min)

        if self._fs_validate():
            print('Feature scaling applied succesfully\n', 'All values in range 0-1')
        else: print('Feature scaling failed')

    def create_model(self, l_rate, epochs):
        # Initialized weights
        weights = [0 for i in range(self.xtrain.shape[1] + 1)]
        for epoch in range(epochs):
            sum_error = 0
            for i in range(len(self.xtrain)):
                yhat  = self._predict(self.xtrain[i], weights)
                error = yhat - self.ytrain[i]
                sum_error += error**2
                weights[0] = weights[0] - l_rate * error
                for j in range(1, len(weights)):
                    weights[j] = weights[j] - l_rate * error * self.xtrain[i][j-1]

        print('Coefficient/weights learned - Model created\n Error: {}\n Weights: {}'.format(sum_error, weights))
        self.weights = np.array(weights)
        return weights

    def test_model(self):
        predictions = []
        for i in range(self.xtest.shape[0]):
            yhat = self._predict(self.xtest[i], self.weights)
            predictions.append(yhat)
        print('Accuracy: {}'.format(self._accuracy(predictions)))



LR = LinearRegression()
LR.feature_scaling()
LR.create_model(0.0112, 40)
LR.test_model()



