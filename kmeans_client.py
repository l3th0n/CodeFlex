import numpy as np

class KMeans_Client:
    # KMeans client inspired by Chris Piech's version (Which is based on a handout by Andrew Ng)
    # Source: http://stanford.edu/~cpiech/cs221/handouts/kmeans.html

    def __init__(self, k):
        self.k         = k
        self.data      = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
        self.centroids = np.array([self.data[i] for i in range(self.k)])
        self.labels    = None

    def get_variables(self):
        return self.k, self.data, self.centroids, self.labels

    def run_clustering(self):
        cnt = 0
        prev_centroids, centroids = np.array([]), self.centroids

        while self.continue_(prev_centroids, centroids):
            prev_centroids = centroids
            cnt += 1
            labels = self.update_labels(centroids)
            centroids = self.update_centroids(labels)

        self.centroids = centroids
        self.labels    = labels

    def update_labels(self, centroids):
        sim1_mat = np.sum(np.sqrt((self.data - centroids[0])**2), axis=1)
        sim2_mat = np.sum(np.sqrt((self.data - centroids[1])**2), axis=1)
        labels = (sim1_mat > sim2_mat).astype(int)
        return labels
            
    def update_centroids(self, labels):
        centroid1 = np.average(self.data[np.where(labels == 0)], 0)
        centroid2 = np.average(self.data[np.where(labels == 1)], 0)
        return np.array([centroid1, centroid2])

    def continue_(self, prev_centroids, centroids):
        return not np.array_equal(prev_centroids, centroids)


client = KMeans_Client(2)
client.run_clustering()
print(client.get_variables()[2])


