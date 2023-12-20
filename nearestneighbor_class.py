import pandas as pd
import random
import math

def main():
	x_data = 'mnist-x.data'
	y_data = 'mnist-y.data'
	P = Perceptron(x_data, y_data, 3, 5)
	P.data_info()
	# P.run()
	P.test()

class Perceptron():
	
	def __init__(self, x_data, y_data, num1=0, num2=1):
		self.y_data = pd.read_csv(y_data, header=None)
		self.y_data = self.y_data[(self.y_data == num1) | (self.y_data == num2)].dropna()
		self.y_data = self.y_data.where(self.y_data == num1, -1)
		self.y_data = self.y_data.where(self.y_data == -1, 1)
		self.x_data = pd.read_csv(x_data, header=None)
		self.x_data = self.x_data.iloc[self.y_data.index.tolist()]

		
		shape = self.y_data.shape
		self.cutoff = int(shape[0] * (5/6))
		self.train_x = self.x_data[:self.cutoff]
		self.train_y = self.y_data[:self.cutoff]
		self.test_x = self.x_data[self.cutoff:]
		self.test_y = self.y_data[self.cutoff:]

		self.weights = []

	def data_info(self):
		print('Train_x = {}'.format(self.train_x.shape))
		print('Train_y = {}'.format(self.train_y.shape))
		print('Test_x = {}'.format(self.test_x.shape))
		print('Test_y = {}'.format(self.test_y.shape))
	
	def test(self):
		cnt = 0
		# print(self.test_x)
		for i, vec1 in self.test_x.iterrows():
			print(self.test_x.ix[i])
			# best_match = None
			# for j, vec2 in enumerate(self.train_x):
			# 	dist = self._distance(vec1, vec2)
			# 	if dist < best_match or not best_match:
			# 		best_match = self.train_y[j]
			# if best_match == self.test_y[i]:
			# 	cnt += 1

		print("Test Accuracy:", cnt / self.test_x.shape[0])

	def _distance(self, vec1, vec2):
		return math.sqrt(sum([(vec1[i] - vec2[i])**2 for i in range(len(vec1))]))


if __name__ == '__main__':
	main()

	# df[(df == 3) | (df == 5)]
