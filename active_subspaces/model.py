""" Model class"""

import numpy as np



class GlobalLinearModel:
	def __init__(self):
		self.a = None	
		self.b = None


	def fit(self, X, fval):
		""" Fit a linear model to measurments

		Parameters
		----------
		X: numpy array of shape [n_samples, n_parameters]
			Location where the function was evaluated

		fval: numpy array of shape [n_samples] or [n_samples, 1]
			Values of the function evaluated at each X[i], i.e.,
			f(X[i]) = fval[i]
		"""

		Y = np.hstack((X, np.ones((X.shape[0],1))))
		c = np.linalg.lstsq(Y,fval)[0]
		self.a = c[-1]
		self.b = c[0:-1]

	def predict(self, X):
		return self.a + np.dot(X, self.b)
