"""Utility for estimating the noise in a function at a point"""
from __future__ import division
import numpy as np
from simrunners import SimulationRunner
import warnings


class SmallStepWarning(UserWarning):
	def __init__(self, h, mesg = ''):
		Exception.__init__(self, "Stepsize too small, recommend increasing h by %gx. %s" % (h,mesg))
		self.h = h

class LargeStepWarning(UserWarning):
	def __init__(self, h, mesg = ''):
		Exception.__init__(self, "Stepsize too large, recommend decreasing h by %gx. %s" % (h,mesg))
		self.h = h
	


def ecnoise(fval):
	"""Estimate the noise present in a function
	
	This is a python reimplementation of ECNoise.m by More and Wild
	avalible from
		
		http://www.mcs.anl.gov/~wild/cnoise/

	The function takes an array of function values taken at uniformly spaced
	points, e.g., 

		f(x - 2h), f(x - h), f(x), f(x + h), f(x + 2h)

	and uses these to estimate the noise in the function f through repeated 
	finite differences to remove local changes.


	The primary difference between this Python implementation and the 
	original ECNoise.m is the use of warnings to advise the user on changing 
	the stepsize in the function evaluations.

	Note this can (should) be used in conjunction with the driver estimate_noise
	which automatically applies the advice recommended by this routine.


	Parameters
	----------
	fval: array like
		Array of function evaluations from which to estimate the noise in the function

	Returns
	-------
	noise: float
		Estimate of the noise in the function
	"""


	fval = np.array(fval).flatten()
	n = len(fval)
	assert n >= 4, "There must be at least four function values to proceed"

	# Step 1: Check that the range of function values is sufficiently small
	fmin, fmax = min(fval), max(fval)

	# This may trigger a divide by zero warning, so we ignore it, being OK with infinite values
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', RuntimeWarning)
		# np.errstate(divide = 'ignore', invalid = 'ignore' ):
		norm_range = (fmax - fmin)/max(abs(fmax), abs(fmin))

	if norm_range > 1.:
		warnings.warn(LargeStepWarning(100, '(normalized range check)'))
		# Although warning, we go ahead and continue to estimate the noise, although this
		# will be contaninated by large fluctuations

	# Step 2: If half the function values are equal, we don't have enough information
	if np.sum(np.diff(fval) == 0) >= n/2.:
		warnings.warn(SmallStepWarning(100, '(too many function values equal)'))
		return 0.
	

	# Step 3: Construct the finite difference table
	DF_table = np.nan * np.ones((n, n), dtype = np.float)
	DF_table[:,0] = fval
	gamma = 1.
	noise_at_level = np.zeros((n,), dtype = np.float)
	sign_change = np.zeros((n,), dtype = np.bool)
	for i in range(1,n):
		gamma *= 0.5*(i/(2.*i-1))
		for j in range(n - i):
			DF_table[j,i] = DF_table[j+1,i-1] - DF_table[j,i-1]

		noise_at_level[i] = np.sqrt(gamma*np.mean(DF_table[:n-i, i]**2))
		emin = np.min(DF_table[:n-i,i])
		emax = np.max(DF_table[:n-i,i])
		sign_change[i] = (emin * emax < 0)
	
	# Step 4: Determine which level of noise to use
	for k in range(n - 3):
		emin = min(noise_at_level[k:k+2])
		emax = max(noise_at_level[k:k+2])
		if (emax < 4*emin and sign_change[k]):
			return noise_at_level[k]

	# Step 5: If we couldn't find a good level, issue a warning and punt on the noise level
	warnings.warn(LargeStepWarning(100, '(Could not find good place to evaluate on the finite difference table)'))	
	return noise_at_level[-1]


#def estimate_noise(f, x, p = None, nf = 9, h = 1e-2, max_recursion = 50, previous_h = None): 
#	"""Estimate the noise in a function near x in direction p 
#
#	This code follows the work of More and Wild, in particular, 
#	the function ECNoise.m, avalible from:
#
#		http://www.mcs.anl.gov/~wild/cnoise/
#
#	The primary change has been to incorporate the advice on scaling the step size
#	h so that an accurate estimation has been made.
#
#	TODO: How do we ensure we do not step outside the domain of the function?
#
#	Parameters:
#	----------	
#		f: SimulationRunner
#			Function whose noise we are attempting to estimate
#		x: np.ndarray
#			Coordinates where to evaluate the function
#		p: np.ndarray
#			Direction in which to evaluate the function (optional)
#		nf: int
#			Number of function evaluations to perform (default 9)
#		h: float
#			Starting stepsize
#		max_recursion : int
#			Maximum number or recursions allowed. This is used internally
#			to prevent infinite recursion.
#
#	Returns:
#	-------	
#		sigma: float
#			Estimation of the standard deviation of the nosie	
#	"""
#
#
#	if previous_h is None:
#		previous_h = [h]
#	else:
#		previous_h.append(h)
#
#	n = x.shape[0]
#	# The x we use internally needs to be a vector, not a n x 1 matrix.
#	if len(x.shape) > 1 and x.shape[1] == 1:
#		x = x.flatten()
#
#	if p is None:
#		# Construct a random direction on the unit sphere
#		p = np.random.randn(n)
#		p /= np.linalg.norm(p)
#
#	# Compute the points on which to evaluate the function
#	X = np.zeros((nf, n))
#	for i, delta in enumerate(np.linspace(-h/2., h/2., num = nf)):
#		X[i,:] = x + delta * p
#
#	# Check if these points are inside the domain:
#	inside = [f.isinside(x_) for x_ in X]
#	if not all(inside):
#		#print "outside domain, h = ", h
#		#print "inside: ", inside
#		# Since we don't actually compute function values for these invalid h, we don't add them to the record
#		if h/2. not in previous_h and len(previous_h) < max_recursion:
#			return estimate_noise(f, x, p = p , nf = nf, h = h/2., max_recursion = max_recursion, previous_h = previous_h)
#		else:
#			raise Exception('No valid domain')
#	
#	# Run the simulation
#	#print h 
#	F = f.run(X)
#	print h, ":", F.flatten()
#
#	# Check that the range of function values is sufficiently small
#	fmin, fmax = min(F), max(F)
#	# This may trigger a divide by zero warning, so we ignore it, being OK with infinite values
#	with warnings.catch_warnings():
#		warnings.simplefilter('ignore', RuntimeWarning)
#		# np.errstate(divide = 'ignore', invalid = 'ignore' ):
#		norm_range = (fmax - fmin)/max(abs(fmax), abs(fmin))
#	if norm_range > 1. and len(previous_h) < max_recursion:
#		# In this case More and Wild consider that noise has not been detected (inform=3)
#		# and recommend retrying with h = h/100
#		if h/100. not in previous_h:
#			print "h too large, re-running with smaller h. ", "new h = ", h/100.
#			return estimate_noise(f, x, p = p, nf = nf, h = h/100., 
#						max_recursion = max_recursion, previous_h = previous_h)
#	
#	# h is too small if half the function values are equal
#	if np.sum(np.diff(F.flatten()) == 0) >= nf/2. and max_recursion > 0:
#		if h*100. not in previous_h:
#			print "h too small, re-running with larger h. ", "new h = ", h*100.
#			return estimate_noise(f, x, p = p, nf = nf, h = h*100., 
#				max_recursion = max_recursion, previous_h = previous_h)
#
#
#	# Construct the finite difference table
#	DF_table = np.nan * np.ones((nf, nf), dtype = np.float)
#	DF_table[:,0] = F.flatten()
#	gamma = 1.		
#	noise_at_level = np.zeros((nf,), dtype = np.float)
#	sign_change = np.zeros((nf,), dtype = np.bool)
#	for i in range(1,nf):
#		gamma *= 0.5*(i/(2.*i-1))
#		for j in range(nf - i):
#			DF_table[j,i] = DF_table[j+1,i-1] - DF_table[j,i-1]
#
#
#		noise_at_level[i] = np.sqrt(gamma*np.mean(DF_table[:nf-i, i]**2))
#		emin = np.min(DF_table[:nf-i,i])
#		emax = np.max(DF_table[:nf-i,i])
#		sign_change[i] = (emin * emax < 0)
#
#	
#	# Determine which level of noise to use
#	for k in range(nf - 3):
#		emin = min(noise_at_level[k:k+2])
#		emax = max(noise_at_level[k:k+2])
#		if (emax < 4*emin and sign_change[k]):
#			return noise_at_level[k]
#
#	# If none works, shrink the interval 
#	if len(previous_h) < max_recursion:
#		print "h too large, re-running with smaller h (post-eval check). ", "new h = ", h/10.
#		return estimate_noise(f, x, p = p, nf = 2*nf, h = h/10., max_recursion = max_recursion, 
#				previous_h = previous_h)	
#	else:
#		raise StandardError('Could not find an appropreate step size for the More-Wild algorithm')
#		print DF_table
#		print sign_change
#		print noise_at_level
		
	
def estimate_noise(f, x, p = None, nf = 9, h = 1e-2, maxiter = 5): 
	"""Estimate the noise in a function near x in direction p 

	This code follows the work of More and Wild, in particular, 
	the function ECNoise.m, avalible from:

		http://www.mcs.anl.gov/~wild/cnoise/

	The primary change has been to incorporate the advice on scaling the step size
	h so that an accurate estimation has been made.

	TODO: How do we ensure we do not step outside the domain of the function?

	Parameters:
	----------	
		f: SimulationRunner
			Function whose noise we are attempting to estimate
		x: np.ndarray
			Coordinates where to evaluate the function
		p: np.ndarray
			Direction in which to evaluate the function (optional)
		nf: int
			Number of function evaluations to perform (default 9)
		h: float
			Starting stepsize
		maxiter : int
			Maximum number of attempts allowed. This is used internally
			to prevent infinite recursion.

	Returns:
	-------	
		sigma: float
			Estimation of the standard deviation of the nosie	
	"""
	

	n = x.shape[0]
	# The x we use internally needs to be a vector, not a n x 1 matrix.
	if len(x.shape) > 1 and x.shape[1] == 1:
		x = x.flatten()

	if p is None:
		# Construct a random direction on the unit sphere
		p = np.random.randn(n)
		p /= np.linalg.norm(p)

	
	X = np.zeros((nf, n))

	noise_estimate = {}
	for it in range(maxiter):
		# Step 1: Check that all points are inside the domain
		for it2 in range(100):
			# Construct the points where we wish to sample
			for i, delta in enumerate(np.linspace(-h/2., h/2., num = nf)):
				X[i,:] = x + delta * p

			inside = [f.isinside(x_) for x_ in X]
			if all(inside):
				break
			else:
				h = h/2.

		# Now run the simulation
		fval = f.run(X)	
		print h, ' : ', fval
		# If not all points are valid, shrink the domain
		if not np.all(np.isfinite(fval)):
			h = h/10.
		
		else:
			with warnings.catch_warnings(record = True) as w:
				warnings.simplefilter("always", LargeStepWarning) 
				warnings.simplefilter("always", SmallStepWarning) 

				noise = ecnoise(fval)
				noise_estimate[h] = noise
				if len(w) == 0:
					return noise
				elif isinstance(w[0].message, LargeStepWarning):
					#h /= w[0].message.h
					h /= 10.
				elif isinstance(w[0].message, SmallStepWarning):
					h *= 10.
					#h *= w[0].message.h

	# If everything else has failed, return the largest estimate
	print noise_estimate
	return max([noise_estimate[h] for h in noise_estimate])



def estimate_second_derivative(f, x, p, noise_estimate = None, **kwargs):
	""" Estimate the second derivative.

	This implements Algorithm 5.1 from a paper of More and Wild [1] on
	estimating the magnitude of the second derivative based.  This algorithm
	makes use of the noise estimate provided by ECNoise and then evaluates the 
	function at the locations necessary to estimate the second derivative.
	

	[1] Jorge J. More and Stefan M. Wild, "Estimating Derivatives of Noisy
	Simulations", ACM Transactions on Mathematical Software, Volume 38, Article 19,
	April 2012
	

	"""

	tau1 = 100.
	tau2 = 0.1

	# If not provided, estimate the noise
	if noise_estimate is None:
		noise_estimate = estimate_noise(f, x, p, **kwargs)
	

	# First attempt to estimate a good step length
	h_a = noise_estimate**(0.25)
	X = np.array([x + h * p for h in [-h_a,0,h_a]])
	fval = f(X)

	# Estimate the second derivative
	delta_a = (fval[0] - 2 * fval[1] + fval[2])
	mu_a = delta_a/h_a**2
	
	# If this estimate passes the heuristics, return it
	print "mu_a", mu_a
	print "delta_a", delta_a
	print "18a:", abs(delta_a)/noise_estimate, ">= ", tau1
	print "18b:", abs(fval[0] - fval[1]), "<=", tau2*max([abs(fval[0]), abs(fval[1])])
	print "18b:", abs(fval[2] - fval[1]), "<=", tau2*max([abs(fval[2]), abs(fval[1])])
	if abs(delta_a) / noise_estimate >= tau1 and \
		abs(fval[0] - fval[1]) <= tau2 * max([abs(fval[0]), abs(fval[1])]) and \
		abs(fval[2] - fval[1]) <= tau2 * max([abs(fval[2]), abs(fval[1])]):
		return mu_a

	# Try a new step size if this fails
	h_b = (noise_estimate / abs(mu_a))**(0.25)
	# Compute new function values with this step
	fval_b = f(np.array([x + h * p for h in [-h_b, h_b]]))
	fval[0] = fval_b[0]
	fval[2] = fval_b[1]
	
	# Estimate the second derivative
	delta_b = (fval[0] - 2 * fval[1] + fval[2])
	mu_b = delta_b/h_b**2
	print "mu_b", mu_b
	print "delta_b", delta_b
	print "18a:", abs(delta_b)/noise_estimate, ">= ", tau1
	print "18b:", abs(fval[0] - fval[1]), "<=", tau2*max([abs(fval[0]), abs(fval[1])])
	print "18b:", abs(fval[2] - fval[1]), "<=", tau2*max([abs(fval[2]), abs(fval[1])])
	# If this estimate passes the heuristics, return it
	if abs(delta_b) / noise_estimate >= tau1 and \
		abs(fval[0] - fval[1]) <= tau2 * max([abs(fval[0]), abs(fval[1])]) and \
		abs(fval[2] - fval[1]) <= tau2 * max([abs(fval[2]), abs(fval[1])]):
		return mu_b
	
	# Otherwise, see if we can return mu_b
	if abs(abs(mu_a) - abs(mu_b)) <= 0.5 * abs(mu_b):
		return mu_b
	print mu_a, mu_b
	raise Exception("Could not estimate second derivative")

	




def estimate_stepsize(f, x, p, noise_estimate = None, second_der_bound = None, **kwargs):
	""" Estimate the 'optimum' stepsize for forward difference

	"""

	if noise_estimate is None:
		noise_estimate = estimate_noise(f, x, p, **kwargs)

	if second_der_bound is None:
		second_der_bound = abs(estimate_second_derivative(f, x, p, noise_estimate = noise_estimate))

	return 8**(.25)*float(noise_estimate/second_der_bound)**(0.5)  
