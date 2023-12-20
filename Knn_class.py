from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

datatrain = np.loadtxt('IDSWeedCropTrain.csv' , delimiter= ',')
datatest = np.loadtxt('IDSWeedCropTest.csv' , delimiter= ',')
xtrain = datatrain[:,:-1]
ytrain = datatrain[:,-1]
# self.xtest = datatest[:,:-1]
# self.ytest = datatest[:,-1]

class Knn:
    # Creates an K-nearest neighbor data structure.

    def __init__(self):
        datatrain = np.loadtxt('IDSWeedCropTrain.csv' , delimiter= ',')
        datatest = np.loadtxt('IDSWeedCropTest.csv' , delimiter= ',')
        self.xtrain = datatrain[:,:-1]
        self.ytrain = datatrain[:,-1]
        self.xtest = datatest[:,:-1]
        self.ytest = datatest[:,-1]

    def train_predict_model(self, k):
        knn_obj = self.train_model(k, self.xtrain, self.ytrain)
        return self.test_model(self.ytest, knn_obj.predict(self.xtest))

    def train_model(self, k, traindata, trainlabel):
        knn_obj = KNeighborsClassifier(n_neighbors=k)
        knn_obj.fit(traindata, trainlabel)
        return knn_obj

    def test_model(self, testlabel, model):
        return accuracy_score(testlabel, model)

    def run_cross_validation(self, folds=5, k_range=[1, 3, 5, 7, 9, 11], shuffle=False):
        # Create dictionary of relevant k's with the value being accuracy.
        acc_dict = {k: 0 for k in k_range} 
        # Create the KFold object; Holds the the amount of folds or train/test datasets.
        # Shuffle on for increased randomness of datapoints.
        cv = KFold(n_splits=folds, shuffle=shuffle)

        for k in acc_dict.keys():

            for train_index, test_index in cv.split(self.xtrain):
                xtrain_cv, xtest_cv, ytrain_cv, ytest_cv = \
                self.xtrain[train_index], self.xtrain[test_index], self.ytrain[train_index], self.ytrain[test_index]

                model = self.train_model(k, xtrain_cv, ytrain_cv)
                acc = self.test_model(ytest_cv, model.predict(xtest_cv))
                acc_dict[k] = acc_dict[k] + acc

        for key, value in acc_dict.items(): acc_dict[key] = round(float(1 - value / folds), 4) # Calculate average accuracy score
        # print(acc_dict)
        return acc_dict

    def normalize(self):
        scaler = preprocessing.StandardScaler().fit(self.xtrain)
        xtrain_n = scaler.transform(self.xtrain)
        xtest_n = scaler.transform(self.xtest)
        for col in np.arange(0, len(self.xtrain.T)):
            self.xtrain.T[col] = xtrain_n.T[col]
            self.xtest.T[col] = xtest_n.T[col]

    def get_kbest(self):
        acc_dict = self.run_cross_validation()
        kbest = min(acc_dict, key=acc_dict.get)
        return kbest

class Knn_selfimplemented:
    # Creates the self implemented knn Classifier
    def __init__(self):
        datatrain = np.loadtxt('IDSWeedCropTrain.csv' , delimiter= ',')
        datatest = np.loadtxt('IDSWeedCropTest.csv' , delimiter= ',')
        self.xtrain = datatrain[:,:-1]
        self.ytrain = datatrain[:,-1]
        self.xtest = datatest[:,:-1]
        self.ytest = datatest[:,-1]

    def normalize(self):
        for col in np.arange(0, len(self.xtrain.T)):
            mean  = np.average(self.xtrain.T[col])
            stdev = np.std(self.xtrain.T[col])
            self.xtrain.T[col] = (self.xtrain.T[col] - mean) / stdev
            self.xtest.T[col] = (self.xtest.T[col] - mean) / stdev

    def euclidian_dist(self, datapoint1, datapoint2):
        # Calculates the euclidian distance between two vectors or datapoints
        dist = 0
        for idx in range(0, len(datapoint1)):
            dist += (datapoint1[idx] - datapoint2[idx]) ** 2
        return np.sqrt(dist)

    def best_neighbors(self, datapoint, dataset,  k):
        # Returns a list with k number of closest neighbors
        dists, dist = [], 0
        for i in range(0, len(dataset)):
            dists.append((self.euclidian_dist(datapoint, dataset[i]), i))
        return [tup[1] for tup in sorted(dists)][:k]

    def calc_label(self, best_neighbors, datalabels):
        labelsum = 0
        for idx in best_neighbors:
            labelsum += datalabels[idx] 
        # print(np.round(labelsum / len(best_neighbors)))
        return np.round(labelsum / len(best_neighbors))

    def get_accuracy(self, ylabels, predicted):
        cnt = 0
        for label1, label2 in zip(predicted, ylabels):
            if label1 == label2: cnt += 1
        return cnt / len(ylabels)

    def run_cross_validation(self, folds=5, k_range=[1, 3, 5, 7, 9, 11], shuffle=False):
        # Create dictionary of relevant k's with the value being accuracy.
        acc_dict = {k: 0 for k in k_range} 
        # Create the KFold object; Holds the the amount of folds or train/test datasets.
        # Shuffle on for increased randomness of datapoints.
        cv = KFold(n_splits=folds, shuffle=shuffle)

        for k in acc_dict.keys():

            for train_index, test_index in cv.split(self.xtrain):
                xtrain_cv, xtest_cv, ytrain_cv, ytest_cv = \
                self.xtrain[train_index], self.xtrain[test_index], self.ytrain[train_index], self.ytrain[test_index]

                predicted = []
                for datapoint in xtest_cv:
                    best_neighbors = self.best_neighbors(datapoint, xtrain_cv, k)
                    predicted.append(self.calc_label(best_neighbors, ytrain_cv))

                acc = self.get_accuracy(ytest_cv, predicted)
                acc_dict[k] = acc_dict[k] + acc

        for key, value in acc_dict.items(): acc_dict[key] = round(float(1 - value / folds), 4) # Calculate average accuracy score
        print(acc_dict)
        return acc_dict

    def train_predict_model(self, k):
        predicted = []
        for datapoint in self.xtest:
            best_neighbors = self.best_neighbors(datapoint, self.xtrain, k)
            predicted.append(self.calc_label(best_neighbors, self.ytrain))

        acc = self.get_accuracy(predicted, self.ytest)
        return acc





obj = Knn()
obj.normalize()
print(obj.run_cross_validation())
print(obj.train_predict_model(3))

obj = Knn_selfimplemented()
obj.normalize()
print(obj.run_cross_validation())
print(obj.train_predict_model(3))



