import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.decomposition import PCA


class PCA_Class:
    def __init__(self, datalabel):
        if datalabel == 'murder': self.data = np.loadtxt('murderdata2d.txt')
        elif datalabel == 'pesticide': self.data = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
        elif datalabel == 'pesticide_test': self.data = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')
        self.eigenvalues  = None
        self.eigenvectors = None
    
    def eigen(self):
        data_norm = self.data - np.average(self.data, axis=0)
        # print(np.average(self.data, axis=0))

        # print(self.data, data_norm)
        cov_mat = data_norm.T @ data_norm
        self.eigenvalues, self.eigenvectors = np.linalg.eig(cov_mat)

    def get_eigen(self):
        return self.eigenvalues, self.eigenvectors

    def scatter(self, plot=True):
        plt.scatter(self.data[:,1], self.data[:,0], c='red')
        if plot: 
            plt.savefig('scatter')
            plt.show()

    def scatter_eigen(self):
        self.scatter(False)
        mean = np.flip(np.average(self.data, axis=0), 0)
        vec1 = self.eigenvectors[0]*np.std(self.data[:,1])
        vec2 = self.eigenvectors[1]*np.std(self.data[:,0])
        
        plt.arrow(*mean, *vec1, head_width=0.5, head_length=0.6)
        plt.arrow(*mean, *vec2, head_width=0.5, head_length=0.6)
        plt.axis('equal')
        plt.grid()
        plt.xlabel('Murders per 1,000,000 inhabitants (#)')
        plt.ylabel('Unemployement procent (%)')
        plt.title('Murders vs Unemployement')
        plt.savefig('scatter_eigen')
        plt.show()





    


client = PCA_Class('murder')
client.eigen()
print(client.get_eigen())
