"""Base domain types"""

import numpy as np
from utils.doc_inherit import doc_inherit
from utils.qp_solver import QPSolver



class Domain(object):
	
	def sample(self, x = None):
		""" Generate a random sample from the domain
		
		Parameters
		----------
		x : None or np.ndarray(m)
			Sample points x that match every not NaN entry of x
		"""
		
	def isinside(self, x):
		"""Determines if a point is inside the domain

		"""
		raise NotImplementedError

#	def distance_to_boundary(self, x, p):
#		"""Compute the distance alpha such that x + alpha * p is on the boundary of the domain
#
#		Parameters
#		----------
#		x : np.ndarray(m)
#			Starting point in the domain
#
#		p : np.ndarray(m)
#			Direction from p in which to head towards the boundary
#
#		Returns
#		-------
#		alpha: float
#			Distance to boundary along direction p
#		"""
#		raise NotImplementedError


class BoxDomain(Domain):
	
	def __init__(self,lb, ub ):
		m = lb.shape[0]
		assert m == ub.shape[0], "lb and ub must be the same size"

		self.lb = lb
		self.ub = ub

	@doc_inherit	
	def sample(self, x = None):

		x_sample = np.random.uniform(self.lb, self.ub)
		if x is None:
			return x_sample
		else:
			# Replace those points 
			I = np.isnan(x)
			x_sample[I] = x[I]
		return x_sample

	@doc_inherit	
	def isinside(self, x):
		if np.any(x < self.lb) or np.any(x > self.ub):
			return False
		else:
			return True

class LinIneqDomain(Domain):
	"""Defines a domain specified by a linear inequality constraint

	This defines a domain 

		{x: A x <= b and lb <= x <= ub }

	"""
	
	def __init__(self, A, b, lb = None, ub = None):
		m, n = A.shape

		self.A = np.copy(A)
		self.b = np.copy(b)

		# Add the additional bound constraints into the matrix
		if lb is not None:
			I = np.eye(n)
			rows = ~np.isnan(lb)
			self.A = np.vstack([self.A, -I[rows]])
			self.b = np.hstack([self.b, -lb[rows]])

		if ub is not None:
			I = np.eye(n)
			rows = ~np.isnan(ub)
			self.A = np.vstack([self.A, I[rows]])
			self.b = np.hstack([self.b, ub[rows]])

		m, n = self.A.shape
		print self.b
		# Determine the centroid 

		# get an initial feasible point using the Chebyshev center. 
		normA = np.sqrt( np.sum( np.power(self.A, 2), axis=1 ) ).reshape((m, 1))
		AA = np.hstack(( self.A, normA ))
		c = np.zeros((n+1, 1))
		c[-1] = -1.0

		qps = QPSolver()
		zc = qps.linear_program_ineq(c, -AA, -np.copy(self.b))
		
		self.centroid = zc[:-1].reshape((n, 1))
		self.z0 = np.copy(self.centroid)


	def isinside(self, x):
		return np.all(np.dot(self.A, x) <= self.b)	
	
	def sample(self, x = None):
		m, n = self.A.shape


		maxiter = 500

		if x is not None:
			raise NotImplementedError
		

		else:
			# Attempt to move the center
			ztol = 1e-6
			eps0 = ztol/4.0

			# First attempt to find a valid search direction from the current
			# location z0, 
			bad_dir = True
			for it in range(maxiter):
				d = np.random.normal(size = (n,1))
				# If we have a valid search direction, stop
				if np.all(np.dot(self.A, self.z0 + eps0*d) < self.b):
					bad_dir = False
					break

			# If that fails, restart at the centroid
			if bad_dir is True:
				self.z0 = np.copy(self.centroid)
				for it in range(maxiter):
					d = np.random.normal(size = (n,1))
					# If we have a valid search direction, stop
					if np.all(np.dot(self.A, self.z0 + eps0*d) < self.b):
						bad_dir = False
						break

				# If still haven't found a good direction, error out
				if bad_dir is True:
					raise Exception('could not find a good direction')
			
		
			# Now pick a length along that direction
			f = self.b - np.dot(self.A, self.z0).flatten()
			g = np.dot(self.A, d).flatten()
			
			# find an upper bound on the step
			min_ind = ((g >= 0 ) & ( f > np.sqrt(np.finfo(np.float).eps))).flatten()
			eps_max = np.amin(f[min_ind]/g[min_ind])

			# find a lower bound on the step
			max_ind = np.logical_and(g<0, f > np.sqrt(np.finfo(np.float).eps)).flatten()
			eps_min = np.amax(f[max_ind]/g[max_ind])

			step_length = np.random.uniform(eps_min, eps_max)
	
			# update the current location
			self.z0 += step_length * d
			
			return np.copy(self.z0)

class GaussianDomain(Domain):
	def __init__(self, mean, covariance = None):
		pass
