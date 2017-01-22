"""Base domain types"""

import numpy as np
from copy import deepcopy
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
		raise NotImplementedError
		
	def isinside(self, x):
		"""Determines if a point is inside the domain

		"""
		raise NotImplementedError


	def extent(self, x, p):
		"""Compute the distance alpha such that x + alpha * p is on the boundary of the domain

		Parameters
		----------
		x : np.ndarray(m)
			Starting point in the domain

		p : np.ndarray(m)
			Direction from p in which to head towards the boundary

		Returns
		-------
		alpha: float
			Distance to boundary along direction p
		"""
		raise NotImplementedError

	def normalize(self, x):
		raise NotImplementedError
	
	def unnormalize(self, x):
		raise NotImplementedError

	def __add__(self, other):
		assert isinstance(other, Domain)
		if isinstance(other, ComboDomain):
			ret = deepcopy(other)
			ret.domains.insert(0, deepcopy(self))
		else:
			ret = ComboDomain()
			ret.domains = [deepcopy(self), deepcopy(other)]
		return ret
	
	def __radd__(self, other):
		assert isinstance(other, Domain)
		if isinstance(other, ComboDomain):
			ret = deepcopy(other)
			ret.domains.append(deepcopy(self))
		else:
			ret = ComboDomain()
			ret.domains = [deepcopy(other), deepcopy(self)]
		return ret


class ComboDomain(Domain):
	def __init__(self, domains):
		self.domains = deepcopy(domains)

	def __add__(self, other):
		ret = deepcopy(self)
		assert isinstance(other, Domain)
		if isinstance(other, ComboDomain):
			ret.domains.extend(deepcopy(other.domains))
		else:
			ret.domains.append(deepcopy(other))
		return ret	

	def __radd__(self, other):
		assert isinstance(other, Domain)
		if isinstance(other, ComboDomain):
			ret = deepcopy(other)
			ret.domains.extend(deepcopy(self.domains))
		else:
			ret = deepcopy(self)
			ret.domains.insert(0,deepcopy(other))
		return ret

	def _split(self, x):
		start = 0
		stop = 0
		x_vec = []
		for dom in self.domains:
			stop += len(dom)
			#print start, stop
			x_vec.append(x[start:stop])
			start = stop
		return x_vec


	def sample(self):
		x = []
		for dom in self.domains:
			x.append(dom.sample())
		return np.hstack(x)

	def isinside(self, x):
		start = 0
		stop = 0
		answer = True
		for dom in self.domains:
			stop += len(dom)
			if not dom.isinside(x[start:stop]):
				return False
			start = stop
		return True

	def extent(self, x, p):
		alpha = [dom.extent(xx, pp) for dom, xx, pp in zip(self.domains, self._split(x), self._split(p))]
		return min(alpha)

	def normalize(self, x):
		y = []
		start = 0
		stop = 0
		for dom in self.domains:
			stop += len(dom)
			y.append(dom.normalize(x[start:stop]))
			start = stop
		return np.hstack(y)	

	def unnormalize(self, x):
		y = []
		start = 0
		stop = 0
		for dom in self.domains:
			stop += len(dom)
			y.append(dom.unnormalize(x[start:stop]))
			start = stop
		return np.hstack(y)	

	def __len__(self):
		return sum([len(dom) for dom in self.domains])

class BoxDomain(Domain):
	
	def __init__(self, lb, ub, repeat = 1):
		"""Uniform Sampling on a Box
		repeat : if scalar domain, 
		"""
		lb = np.array(lb).reshape(-1)
		ub = np.array(ub).reshape(-1)
		assert lb.shape[0] == ub.shape[0], "lower and upper bounds must have the same length"

		self.lb = lb
		self.ub = ub
		if repeat > 1:
			assert lb.shape[0] == 1, "Can only repeat with one dimensional domains" 
		self.repeat = repeat

	def __len__(self):
		return self.lb.shape[0]*self.repeat

	#@doc_inherit	
	def sample(self, x = None):

		x_sample = np.random.uniform(self.lb, self.ub)
		if x is not None:
			# Replace those points 
			I = np.isnan(x)
			x_sample[I] = x[I]
		return np.repeat(x_sample, self.repeat)

	#@doc_inherit	
	def isinside(self, x):
		if len(x.shape) == 1:
			if self.repeat > 1:
				if np.any(x[0] != x):
					return False
			if np.any(x < self.lb) or np.any(x > self.ub):
				return False
			return True
		elif len(x.shape) == 2:
			raise NotImplementedError

	#@doc_inherit	
	def normalize(self, X):
		"""
		Return points shifted and scaled to [-1,1]^m.
		"""
		lb = self.lb.reshape((1, lb.shape[0]))
		ub = self.ub.reshape((1, ub.shape[0]))
		return 2.0 * (X - lb) / (ub - lb) - 1.0

	#@doc_inherit	
	def unnormalize(self, X):
		"""
		Return points shifted and scaled to (lb, ub).
		"""
		lb = self.lb.reshape((1, lb.shape[0]))
		ub = self.ub.reshape((1, ub.shape[0]))
		return (ub - lb) * (X + 1.0) / 2.0 + lb

	def extent(self, x, p):
		alpha = float('inf')
		# Now check box constraints
		I = np.nonzero(p)
		y = (self.ub - x)[I]/p[I]
		if np.sum(y>0) > 0:
			alpha = min(alpha, np.min(y[y>0]))	
	
		y = (self.lb - x)[I]/p[I]
		if np.sum(y>0) > 0:
			alpha = min(alpha, np.min(y[y>0]))

		# If on the boundary, the direction needs to point inside the domain
		if np.any(p[self.lb == x] < 0):
			alpha = 0
		if np.any(p[self.ub == x] > 0):
			alpha = 0	
		return alpha
		

class UniformDomain(BoxDomain):
	pass

class NormalDomain(Domain):
	def __init__(self, mean, cov = None):
		self.mean = np.array(mean)
		m = self.mean.shape[0]
		if cov is None:
			cov = np.eye(m)
		self.cov = np.array(cov)
		self.ew, self.ev = np.linalg.eigh(cov)
		assert np.all(self.ew > 0), 'covariance matrix must be positive definite'

	def isinside(self, x):
		if len(x.shape) == 1:
			return True		
		elif len(x.shape) == 2:
			return np.ones(x.shape[0], dtype = np.bool)

	def extent(self, x, p):
		return float('inf')
	
	def normalize(self, x):
		if len(x.shape) == 1:
			return np.dot(self.ev, np.dot(np.diag(1./np.sqrt(self.ew)), x - self.mean))
		elif len(x.shape) == 2:
			return np.dot(self.ev, np.dot(np.diag(1./np.sqrt(self.ew)), x.T - self.mean.reshape(-1,1))).T
		raise NotImplementedError

	def unnormalize(self, y):
		if len(y.shape) == 1:
			return self.mean + np.dot(np.diag(np.sqrt(self.ew)), np.dot(self.ev.T, y))
		elif len(y.shape) == 2:
			return (self.mean.reshape(-1,1) + np.dot(np.diag(np.sqrt(self.ew)), np.dot(self.ev.T, y.T))).T
		raise NotImplementedError

	def sample(self):
		return np.random.multivariate_normal(self.mean, self.cov)
	
	def __len__(self):
		return self.mean.shape[0]

class LogNormalDomain(Domain):
	def __init__(self, mean, cov = None, scaling = 1.):
		if isinstance(mean, float) or isinstance(mean, int):
			mean = [mean]
		if isinstance(cov, float) or isinstance(mean, int):
			cov = [[cov]]
		self.mean = np.array(mean)
		m = self.mean.shape[0]
		if cov is None:
			cov = np.eye(m)
	
		self.cov = np.array(cov)
		if self.mean.shape[0] == 1:
			self.cov.reshape(1,1)
		
		self.ew, self.ev = np.linalg.eigh(cov)
		self.scaling = scaling
		assert np.all(self.ew > 0), 'covariance matrix must be positive definite'

	def isinside(self, x):
		if len(x.shape) == 1:
			return np.all(x > 0)
		else:
			return np.array([np.all(xx > 0) for xx in x], dtype = np.bool)

	def extent(self, x, p):
		return float('inf')
	
	def normalize(self, x):
		if len(x.shape) == 1:
			return np.dot(self.ev, np.dot(np.diag(1./np.sqrt(self.ew)), np.log(x/self.scaling) - self.mean))
		elif len(x.shape) == 2:
			return np.dot(self.ev, np.dot(np.diag(1./np.sqrt(self.ew)), np.log(x/self.scaling).T - self.mean.reshape(-1,1))).T
		raise NotImplementedError

	def unnormalize(self, y):
		if len(y.shape) == 1:
			return np.exp(self.mean + np.dot(np.diag(np.sqrt(self.ew)), np.dot(self.ev.T, y)))*self.scaling
		elif len(y.shape) == 2:
			return np.exp(self.mean.reshape(-1,1) + np.dot(np.diag(np.sqrt(self.ew)), np.dot(self.ev.T, y.T))).T*self.scaling
		raise NotImplementedError

	def sample(self):
		return np.exp(np.random.multivariate_normal(self.mean, self.cov))*self.scaling

	def __len__(self):
		return self.mean.shape[0]

class LinIneqDomain(Domain):
	"""Defines a domain specified by a linear inequality constraint

	This defines a domain 

		{x: A x <= b and lb <= x <= ub }

	"""
	
	def __init__(self, A, b, lb = None, ub = None, center = None):
		m, n = A.shape

		self.A = np.copy(A)
		self.b = np.copy(b)

		# Add the additional bound constraints into the matrix
		if lb is not None:
			#I = np.diag(1./lb)
			rows = np.isfinite(lb)
			#self.A = np.vstack([self.A, -I[rows]])
			#self.b = np.hstack([self.b, -np.ones(np.sum(rows))])
			self.lb = np.copy(lb)
			self.lb[~rows] = -np.inf
		else:
			self.lb = -np.inf * np.ones(n)

		if ub is not None:
			#I = np.diag(1./ub)
			rows = np.isfinite(ub)
			#self.A = np.vstack([self.A, I[rows]])
			#self.b = np.hstack([self.b, np.ones(np.sum(rows))])
			self.ub = np.copy(ub)
			self.ub[~rows] = np.inf
		else:
			self.ub = np.inf * np.ones(n)

		m, n = self.A.shape
		
		assert n == self.lb.shape[0]
		assert n == self.ub.shape[0]

				
		if center is None:
			# get an initial feasible point using the Chebyshev center. 
			normA = np.sqrt( np.sum( np.power(self.A, 2), axis=1 ) ).reshape((m, 1))
			AA = np.hstack(( self.A, normA ))
			c = np.zeros((n+1, 1))
			c[-1] = -1.0
			
			qps = QPSolver()
			zc = qps.linear_program_ineq(c, -AA, -np.copy(self.b), lb = np.hstack([self.lb,0.]), ub = np.hstack([self.ub, np.inf]))
			center = zc[:-1].reshape((n,))
			center[self.only_bounds] = (self.lb[self.only_bounds] + self.ub[self.only_bounds])/2.

		# Check we are a valid point
		assert np.all(self.lb<= center), "failed LB test"
		assert np.all(self.ub>= center), "failed UB test"
		assert np.all(np.dot(self.A, center) <= self.b)

		self.center = center	
		self.z0 = np.copy(self.center)

	def __len__(self):
		return self.center.shape[0]
			
	def isinside(self, x):
		if len(x.shape) == 1:
			#print np.all(np.dot(self.A, x) <= self.b), np.all(self.lb <= x), np.all(x <= self.ub)
			return np.all(np.dot(self.A, x) <= self.b) and np.all(self.lb <= x) and np.all(x <= self.ub)	


	def extent(self, x, p):
		# positive extent
		y = (self.b - np.dot(self.A, x)	)/np.dot(self.A, p)
		alpha = np.min(y[y>0])
		# Now check box constraints
		I = np.nonzero(p)
		y = (self.ub - x)[I]/p[I]
		if np.sum(y>0) > 0:
			alpha = min(alpha, np.min(y[y>0]))	
	
		y = (self.lb - x)[I]/p[I]
		if np.sum(y>0) > 0:
			alpha = min(alpha, np.min(y[y>0]))


		# If on the boundary, the direction needs to point inside the domain
		if np.any(p[self.lb == x] < 0):
			alpha = 0
		if np.any(p[self.ub == x] > 0):
			alpha = 0	
		return alpha
	
	def sample(self, no_recurse = False):

		#A = self.A[:,~self.only_bounds]
		#m, n = A.shape

		maxiter = 500
		bad_dir = True
		for it in range(maxiter):
			p = np.random.normal(size = self.center.shape)
			p[self.only_bounds] = 0.
			alpha_min = -self.extent(self.z0, -p)
			alpha_max = self.extent(self.z0, p)
			if alpha_max - alpha_min > 1e-10:
				bad_dir = False
				break

		if bad_dir and no_recurse ==  False:
			self.z0 = np.copy(self.center)
			return self.sample(no_recurse = True)

		if bad_dir == True and no_recurse == True:
			raise Exception('could not find a good direction')
		
		# update the center
		step_length = np.random.uniform(alpha_min, alpha_max)
		
		# update the current location
		self.z0 += step_length * p
	
		# Sample randomly on the variables that only have bound constraints	
		z = np.copy(self.z0)
		z[self.only_bounds] = np.random.uniform(self.lb[self.only_bounds], self.ub[self.only_bounds])	
		return z


	
	def normalize(self, X):
		"""
		Return points shifted and scaled to [-1,1]^m.
		"""
		if len(X.shape) == 1:
			X = X.reshape(1,-1)
		lb = self.lb.reshape(1, -1)
		ub = self.ub.reshape(1, -1)
		return 2.0 * (X - lb) / (ub - lb) - 1.0

	def unnormalize(self, X):
		"""
		Return points shifted and scaled to (lb, ub).
		"""
		if len(X.shape) == 1:
			X = X.reshape(1, -1)
		lb = self.lb.reshape(1, -1)
		ub = self.ub.reshape(1, -1)
		return (ub - lb) * (X + 1.0) / 2.0 + lb

