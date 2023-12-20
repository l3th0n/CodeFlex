import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, t


class stat:
    def __init__(self, data):
        self.smo = data[data[:,4] == 1]
        self.non = data[data[:,4] == 0]
        self.all = len(data)
        self.all_age = np.sort(data[:,0])
        self.all_fev1 = np.sort(data[:,1])
        self.smo_n = len(self.smo)
        self.non_n = len(self.non)
        self.smo_mean = np.average(self.smo[:,1])
        self.non_mean = np.average(self.non[:,1])
        self.age_smo_mean = np.mean(self.smo[:,0])
        self.age_non_mean = np.mean(self.non[:,0])
        self.smo_var = np.var(self.smo[:,1])
        self.non_var = np.var(self.non[:,1])
        self.smo_std = np.std(self.smo[:,1])
        self.non_std = np.std(self.non[:,1])

    def print_stats(self):
        print('     Smokers / Non-smokers')
        print('n:        {}   {}'.format(self.smo_n, self.non_n))
        print('Mean:     {} {}'.format(np.round(self.smo_mean, 2), np.round(self.non_mean, 2)))
        print('Mean(age):{} {}'.format(np.round(self.age_smo_mean, 2), np.round(self.age_non_mean, 2)))
        print('Variance: {} {}'.format(np.round(self.smo_var, 2), np.round(self.non_var, 2)))
        print('Std. Dev: {} {}'.format(np.round(self.smo_std, 2), np.round(self.non_std, 2)))

    def mean_fev1(self):
        result = (self.smo_mean, self.non_mean)
        return result

    def box_plot(self):
        plt.boxplot([data[data[:,4] == 1][:,1], data[data[:,4] == 0][:,1]])
        plt.xticks([1,2], ('Smokers', 'Non-smokers'))
        plt.ylabel('FEV1 Measurement (Continuous)')
        plt.savefig('BA.boxplot.png')
        plt.show()

    def hyptest(self, implemented):
        numerator   = (self.smo_var/self.smo_n + self.non_var/self.non_n)**2
        denominator_smo = (((self.smo_var/self.smo_n)**2)/(self.smo_n - 1))
        denominator_non = (((self.non_var/self.non_n)**2)/(self.non_n - 1))

        df = numerator/(denominator_smo + denominator_non)

        if implemented:
            std_delta = np.sqrt((self.smo_std/self.smo_n) + (self.non_std/self.non_n))
            tscore = (self.smo_mean - self.non_mean)/std_delta
            
            p = t.cdf(-tscore, df)+ (1 - t.cdf(tscore, df))
            prnt_statement = 'Self-implemented t-test '

        else:
            (tscore, p) = ttest_ind(self.smo[:,1], self.non[:,1], equal_var=False)
            prnt_statement = 'scipy.stats.ttest_ind function '

        print(prnt_statement)
        print('T-score: {}'.format(tscore))
        print('P-value: {}'.format(p))
        print('Degrees of Freedom: {}'.format(df))
        print('Rejection of null-hypothesis: ', p < 0.05)

    def twod_plot(self):
        x_sum = sum(self.all_age)
        xsqr_sum = sum(self.all_age**2)
        y_sum = sum(self.all_fev1)
        xy_sum = sum(self.all_age * self.all_fev1)
        x_devs = self.all_age - np.average(self.all_age)
        y_devs = self.all_fev1 - np.average(self.all_fev1)
        x_std = np.std(self.all_age)
        y_std = np.std(self.all_fev1)

        a = (((y_sum)*(xsqr_sum))-((x_sum)*(xy_sum)))/((self.all*(xsqr_sum))-(x_sum**2))
        b = ((self.all*(xy_sum))-((x_sum)*(y_sum)))/((self.all*(xsqr_sum))-(x_sum**2))

        r = (1/(self.all-1)) * (sum(x_devs * y_devs) / (x_std * y_std))

        xi = np.arange(0, max(self.all_age) + 1)
        line = b*xi+a

        plt.plot(xi, b*xi+a, line, label='Regression Line')

        for idx in range(len(self.all_age)):
            plt.scatter(self.all_age[idx], self.all_fev1[idx], marker='o', c='blue', alpha=0.25)


        plt.xlabel('Age (Discrete)')
        plt.ylabel('FEV1 Measurement (Continuous)')
        plt.legend(['Subjects', 'Regression Line', "Correlation(r) = "+str(np.round(r, 3))])
        plt.savefig('BA.twod_plot.png')
        plt.show()

    def hist(self):
        plt.hist([self.smo[:,0], self.non[:,0]], self.all_age)
        plt.xlabel('Age (Discrete)')
        plt.ylabel('Count (Discrete)')
        plt.legend(['Smokers', 'Non-smokers'])
        plt.savefig('BA.hist.png')
        plt.show()


data = np.loadtxt('smoking.txt')
Statobj = stat(data)
Statobj.print_stats()
Statobj.mean_fev1()
Statobj.box_plot()
Statobj.hyptest(True)
Statobj.hyptest(False)
Statobj.twod_plot()
Statobj.hist()




