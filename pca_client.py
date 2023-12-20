import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.decomposition import PCA


class PCA_Client:
    def __init__(self, dataset):
        if dataset == 'murder': self.data = np.loadtxt('murderdata2d.txt')
        elif dataset == 'pesticide': self.data = np.loadtxt('IDSWeedCropTrain.csv', delimiter=',')
        elif dataset == 'pesticide_test': self.data = np.loadtxt('IDSWeedCropTest.csv', delimiter=',')
        self.eigenvalues  = None
        self.eigenvectors = None
    
    def eigen(self):
        corr_mat = np.corrcoef(self.data.T)
        cov_mat = np.cov(self.data.T)
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

    def plot_pcs(self):
        pc_number = range(1, len(self.eigenvalues)+1)
        plt.plot(pc_number, self.eigenvalues, color='green', marker='o', linestyle='dashed',
                                                                    linewidth=2, markersize=8)
        plt.xticks(pc_number)
        plt.xlabel('Principal Component (#)')
        plt.ylabel('Variance captured ($\sigma^{2})$')
        plt.title('Principal Component vs Variance')
        plt.savefig('plot_pcs')
        plt.show()

    def plot_pcs_cum(self):
        pc_number = range(1, len(self.eigenvalues)+1)
        var_sum = sum(self.eigenvalues)
        eigenvalues_cum = np.cumsum([var/var_sum for var in self.eigenvalues])

        plt.step(pc_number, eigenvalues_cum, color='green', marker='o', linestyle='dashed',
                                                            linewidth=1, markersize=4)
        plt.plot(pc_number, eigenvalues_cum, linestyle='-', linewidth=2)
        plt.axhline(y=.9, linewidth=1, linestyle='dotted', color='r')
        plt.axhline(y=.95, linewidth=1, linestyle='dotted', color='r')

        plt.grid()
        plt.xticks(pc_number)
        plt.yticks(np.arange(0, 1.0, 0.05))
        plt.xlabel('Principal Component (#)')
        plt.ylabel('Cumulated Variance captured (%)')
        plt.title('Principal Component vs Cumulated Variance')
        plt.savefig('plot_pcs_cum')
        plt.show()

    def mds(self, d):
        pca = PCA(n_components=d)
        pca.fit(self.data)
        X_pca = pca.transform(self.data)
        plt.scatter(X_pca[:,0], X_pca[:,1])
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Principal Component 1 vs Principal Component 1\nProjection of datasample onto subspace')
        plt.savefig('mds')
        plt.show()



    


client = PCA_Client('murder')
client.eigen()
eigenvalues, eigenvectors = client.get_eigen()
# print(eigenvalues, eigenvectors)
# client.scatter()
# client.scatter_eigen()

# client = PCA_Client('pesticide')
# client.eigen()
# client.plot_pcs()
# client.plot_pcs_cum()
# client.mds(2)