import numpy as np
import gzip
#import matplotlib.pyplot as plt

def get_training_images (n: int):
	with gzip.open('dataset/train-images-idx3-ubyte.gz') as f:
		f.read(16)
		buffer = f.read(n * 28 * 28)
		imgs = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32).reshape(n, 28, 28)
		return imgs

def get_training_labels (n: int):
	with gzip.open('dataset/train-labels-idx1-ubyte.gz') as f:
		f.read(8)
		buffer = f.read(n)
		labels = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
		return labels

def get_testing_images (n: int):
	with gzip.open('dataset/t10k-images-idx3-ubyte.gz') as f:
		f.read(16)
		buffer = f.read(n * 28 * 28)
		imgs = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32).reshape(n, 28, 28)
		return imgs

def get_testing_labels (n: int):
	with gzip.open('dataset/t10k-labels-idx1-ubyte.gz') as f:
		f.read(8)
		buffer = f.read(n)
		labels = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
		return labels

def normalize (img):
	return (img - np.mean(img))/np.std(img)

class MNISTClassifier ():

	def __init__ (self):
		self.input = np.zeros(784)
		self.hidden = np.zeros(16)
		self.output = np.zeros(10)

		self.weights = [np.random.rand(784, 16)-0.5, np.random.rand(16, 10)-0.5]

		ReLU = lambda x : x*(x>0)
		self.activation = np.vectorize(ReLU)

	def feedforward (self, x):
		self.input = x
		self.hidden = self.activation(np.dot(self.input,self.weights[0]))
		self.output = self.activation(np.dot(self.hidden,self.weights[1]))
		return self.softmax(self.output)

	def softmax (self, output, index=None):
		if index != None:
			n = output - np.max(output)
			return np.exp(n[0][index])/np.sum(np.exp(n[0]))
		else:
			n = output - np.max(output)
			return np.exp(n)/np.sum(np.exp(n))

	def ReLUDerivative (self, x):
		return np.vectorize(lambda x : x < 0)(x)

	def softmaxDerivative (self, x):
		output = np.zeros((10, 10))

		for (i,j) in zip(range(10), range(10)):
			output[i][j] = np.dot(self.softmax(x, i).T, ((i==j) - self.softmax(x,j)))

		return output

	def loss (self, expected, output):
		return np.sum(np.square(expected-output))

	def backpropagation (self, expected):
		deltaLossY = 2 * (self.output - expected)
		deltaYZ = self.softmaxDerivative(self.output)
		deltaZWOne = np.dot(self.ReLUDerivative(self.hidden), self.hidden.T)
		deltaLossWOne = np.dot(np.dot(deltaYZ, deltaLossY.T), deltaZWOne)
		print(self.ReLUDerivative(self.output).shape, self.weights[1].shape, self.input.shape, self.ReLUDerivative(self.hidden).shape)
		deltaLossWZero = np.dot(np.dot(self.ReLUDerivative(self.output), self.weights[1].T), np.dot(self.input.T, self.ReLUDerivative(self.hidden))).T
		
		self.weights[0] -= deltaLossWZero
		self.weights[1] -= deltaLossWOne


classifier = MNISTClassifier()

NUM_OF_IMGS = 5

losses = []

for (img, label, index) in zip(get_training_images(NUM_OF_IMGS), get_training_labels(NUM_OF_IMGS), range(NUM_OF_IMGS)):
	expected = np.zeros(10)
	expected[int(label)-1] = 1

	loss = classifier.loss(output=classifier.feedforward(normalize(img.reshape(1,784))), expected=expected)
	print(f"Loss: {loss}")
	classifier.backpropagation(expected)

	losses.append(loss)
