import pandas as pd
import random

def main():
	x_data = 'mnist-x.data'
	y_data = 'mnist-y.data'
	P = Perceptron(x_data, y_data, 3, 5)
	P.data_info()
	P.run()
	P.test()

class Perceptron():
	
	def __init__(self, x_data, y_data, num1, num2):
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
	
	def run(self):
		w = self.train_x.shape[1] * [0]
		cnt = 0

		while not self._converged(cnt):
			cnt = 0

			for i in range(self.cutoff):
				x = self.train_x.iloc[i]
				y = int(self.train_y.iloc[i])
				z = sum([w[i] * x[i] for i in range(self.train_x.shape[1])])

				if z >= 0 and y == -1:
					w = w - x
				elif z < 0 and y == 1:
					w = w + x
				else: cnt += 1

		self.weights = w
		return w

	def _converged(self, count):
		print(count,(self.cutoff * 0.95), count / self.cutoff)
		return count >= (self.cutoff * 0.95)

	def test(self):
		cnt = 0
		for i in range(self.test_x.shape[0]):
			x = self.test_x.iloc[i]
			y = int(self.test_y.iloc[i])
			z = sum([self.weights[i] * x[i] for i in range(self.test_x.shape[1])])

			if z >= 0 and y == 1:
				cnt += 1
			elif z < 0 and y == -1:
				cnt += 1

		print("Test Accuracy:", cnt / self.test_x.shape[0])

if __name__ == '__main__':
	main()

	# df[(df == 3) | (df == 5)]
