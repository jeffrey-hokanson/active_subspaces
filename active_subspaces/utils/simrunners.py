"""Utilities for running several simulations at different inputs."""

import numpy as np
import time
from misc import process_inputs
import warnings
import itertools
import functools
# checking to see if system has multiprocessing
try:
	import multiprocessing as mp
	HAS_MP = True
except ImportError, e:
	HAS_MP = False
	pass

try: 
	from celery_runner import celery_runner
	#import marshal
	import dill
	HAS_CELERY = True
except ImportError, e:
	HAS_CELERY = False	

try:
	from progress import Bar
	HAS_PROGRESS = True
except:
	HAS_PROGRESS = False
				
try: 
	from rq import Queue
	from redis import Redis
	HAS_RQ = True
except:
	HAS_RQ = False


class SimulationRunner():
	"""A class for running several simulations at different input values.

	Attributes
	----------
	fun : function 
		runs the simulation for a fixed value of the input parameters, given as
		an ndarray
	
	backend : {'loop', 'multiprocessing', 'celery'}
		Specifies how each evaluation of the function f should be run.
		
		* loop - use a for loop over fun
		* multiprocessing - distribute the function across multiple cores using 
			the multiprocessing library
		* celery - use the Celery distributed task queue to split up function
			evaluations

	domain : Domain
		Specifies the domain of provided function, allowing to check that all
		points that are to be run are inside the domain.

	progress : {True, False}
		If true, displays a progressbar showing how many simulations have been run

	units : {'normalized', 'application'}
		Specifies the units to be used when calling run

	nproc : int
		Number of processes to start when using multiprocessing. If negative,
		use as many processes as tasks

	See Also
	--------
	utils.simrunners.SimulationGradientRunner

	Notes
	-----
	The function fun should take an ndarray of size 1-by-m and return a float.
	This float is the quantity of interest from the simulation. Often, the
	function is a wrapper to a larger simulation code.
	"""

	def __init__(self, fun, backend = None, nproc = None, domain = None,
		progress = False, units = 'application', args = None, kwargs = None,
		path = None, save_file = None, tic = 1., **additional_args):
		"""Initialize a SimulationRunner.

		Parameters
		----------
		fun : function  
			a function that runs the simulation for a fixed value of the input 
			parameters, given as an ndarray. This function returns the quantity 
			of interest from the model. Often, this function is a wrapper to a 
			larger simulation code.

		paths: list of paths
			Specificies the location of python scripts to be run so that celery 
			knows where to look.  This defaults to the current directory when the 
			python script was called.
		"""
		if not hasattr(fun, '__call__'):
			raise TypeError('fun should be a callable function.')

		if path is None:
			import os
			path = [os.getcwd()]
		self.path = path
		self.fun = fun
		self.save_file = save_file
		self.tic = tic
		self.additional_args = additional_args
		# Default backend
		if backend is None:
			backend = 'loop'

		# Check the user has specified a valid backend
		if backend not in ['loop', 'multiprocessing', 'celery', 'rq']:
			raise TypeError('Invalid backend chosen')

		# Check if the backend selected is avalible
		if backend == 'multiprocessing' and HAS_MP is False:
			backend = 'loop'
			warnings.warn('multiprocessing not avalible, defaulting to "loop" backend')
		elif backend == 'celery' and HAS_CELERY is False:
			backend = 'loop'
			warnings.warn('celery not avalible, defaulting to "loop" backend')
		elif backend == 'rq' and HAS_RQ is False:
			backend = 'loop'
			warnings.warn('rq not avalible, defaulting to "loop" backend')

		self.backend = backend

		# Setup the selected backend
		if backend == 'loop':
			self.run = self._run_loop	
		elif backend == 'multiprocessing':
			if nproc is None:
				nproc = mp.cpu_count() - 1
			self.nproc = nproc
			self.run = self._run_multiprocessing
		elif backend == 'celery':
			self.run = self._run_celery
		elif backend == 'rq':
			self.run = self._run_rq

		self.__call__ = self.run

		# Setup the domain options
		self.domain = domain
	
		self.progress = progress and HAS_PROGRESS
		self.units = units
		if self.domain is None:
			self.isinside = lambda x: True
		else:
			if self.units == 'normalized':
				self.isinside = lambda x: self.domain.isinside(self.domain.unnormalize(x).flatten())
			else:
				self.inside = self.domain.isinside


		if args is None:
			args = []
		self.args = args

		if kwargs is None:
			kwargs = {}
		self.kwargs = kwargs


	def _format_output(self, output):
		# Format and store the output
		# We'll need to check the output size and build the matrix appropreately
		M = len(output)
		n_output = None
		for i, out in enumerate(output):
			# Find the dimenison of the output
			if n_output is None and out is not None:
				try:
					out = np.array(out).flatten()
					n_output = out.shape[0]
					F = np.zeros((M,n_output), dtype = out.dtype)
				except:
					pass
			if n_output is not None:
				try:
					F[i] = np.array(out).flatten()
				except: 
					F[i] = np.nan

		# If no evalution was successful
		if n_output is None:
			F = np.nan*np.ones((M,1))
		return F

	def _run_loop(self, X):
		""" Runs a simple for-loop over the target function
		"""
		X, M, m = process_inputs(X)
		if self.units == 'normalized':
			X = self.domain.unnormalize(X)
		# We store the output in a list so that we can handle failures of the function
		# to evaluate
		output = []
		for i in range(M):
			# Try to evaluate the function
			try:
				out = self.fun(X[i,:].reshape((1,m)), *self.args, **self.kwargs)
			except:
				out = None
			output.append(out)

		return self._format_output(output)

	def _run_multiprocessing(self, X):
		X, M, m = process_inputs(X)
		if self.units == 'normalized':
			X = self.domain.unnormalize(X)
		
		nproc = self.nproc
		if nproc < 0:
			nproc = len(X)
		pool = mp.Pool(processes = nproc)
		try:
			if hasattr(self.fun, 'im_class'): 	# If the function is a member of a class
				#TODO Implement kwarg caching for class functions
				arg_list_objects = []
				arg_list_inputs = []
				for i in range(M):
					arg_list_objects.append(self.fun.im_self)
					arg_list_inputs.append(X[i])
				#These are for parallel computation with a class method
				def target(): pass
				def target_star(args): return target(*args)
				target.__code__ = self.fun.im_func.__code__
				output = pool.map(target_star, itertools.izip(arg_list_objects, arg_list_inputs))
			else: 			# Just a plain function
				if len(self.args) > 0 and len(self.kwargs) > 0:
					# Freeze the additional arguments to the function
					# http://stackoverflow.com/a/39366868
					fun = functools.partial(self.fun, *self.args, **self.kwargs)
				elif len(self.args) > 0:
					fun = functools.partial(self.fun, *self.args)
				elif len(self.kwargs) > 0:
					fun = functools.partial(self.fun, **self.kwargs)
				else:
					fun = self.fun

				if HAS_PROGRESS and self.progress:
					bar = Bar(expected_size = len(X))
					result = pool.map_async(fun, X, chunksize = 1)
					start_time = time.time()
					tick = 0.1
					while not result.ready():
						bar.show(len(X) - result._number_left)	
						time.sleep(tick - ((time.time() - start_time) % tick ))
					bar.show(len(X))
					bar.done()
					output = result.get()
				else:
					output = pool.map(fun, X)
		
			pool.close()
			pool.join()
			return self._format_output(output)	
		except:	# If there is a failure in multiprocessing, disable it and restart
			warnings.warn('multiprocessing failed; dropping to "loop" backend')
			self.run = self._run_loop
			self.backend = 'loop'
			return self.run(X)
		
	def _run_celery(self, X):
		X, M, m = process_inputs(X)
		if self.units == 'normalized':
			X = self.domain.unnormalize(X)
		# store the function
		#marshal_func = marshal.dumps(self.fun.func_code)
		marshal_func = dill.dumps(self.fun)
		results = [celery_runner.delay(x, marshal_func, self.args, self.kwargs) for x in X]

		# Time between checking for results
		start_time = time.time()
		if HAS_PROGRESS and self.progress:
			bar = Bar(expected_size = len(results))
		while True:
			# Check if everyone is done
			status = [res.ready() for res in results]
			if HAS_PROGRESS and self.progress:
				bar.show(np.sum(status))

			if all(status):
				break
			else:
				time.sleep(self.tic - ((time.time() - start_time) % self.tic ))

		if HAS_PROGRESS and self.progress:
			bar = bar.done()

		output = []
		for i, res in enumerate(results):
			try:
				output.append(res.get())
			except Exception as e:
				print "result ", i, 'failed with exception', e
				output.append(None)

		return self._format_output(output)
	
	def _run_rq(self, X):
		X, M, m = process_inputs(X)
		if self.units == 'normalized':
			X = self.domain.unnormalize(X)
		# store the function
		#marshal_func = marshal.dumps(self.fun.func_code)
		marshal_func = dill.dumps(self.fun)

		redis_conn = Redis(**self.additional_args)
		q = Queue(connection = redis_conn)


		jobs = [q.enqueue_call(celery_runner, args = [x, marshal_func, self.args, self.kwargs], ttl = 86400) for x in X]
	
		if self.progress:
			bar = Bar(expected_size = len(jobs))
		
		done = False
		start_time = time.time()
		outputs = len(jobs)*[ None ]
		while True:
			#status = [ job.is_finished or job.is_failed for job in jobs]
			#if self.progress:
			#	bar.show(np.sum(status))
			
			# store outputs
			# We need to do this at every iteration as the results are flushed by default after 500 seconds
			for i, job in enumerate(jobs):
				if outputs[i] is None:
					outputs[i] = job.result

			number_done = 0
			for out in outputs:
				if out is not None:
					number_done += 1
			bar.show(number_done)	

			if self.save_file is not None:
				X = self._format_output(outputs)
				np.savetxt(self.save_file, X)
	
			if number_done == len(jobs):
				break
			time.sleep(self.tic - ((time.time() - start_time) % self.tic ))

		if HAS_PROGRESS and self.progress:
			bar = bar.done()

		return self._format_output(outputs)
		

def SimulationGradientRunner(*args, **kwargs):
	"""Evaluates gradients at several input values.
	
	
	A class for running several simulations at different input values that
	return the gradients of the quantity of interest.

	Attributes
	----------
	dfun : function 
		a function that runs the simulation for a fixed value of the input 
		parameters, given as an ndarray. It returns the gradient of the quantity
		of interest at the given input.

	See Also
	--------
	utils.simrunners.SimulationRunner

	Notes
	-----
	The function dfun should take an ndarray of size 1-by-m and return an
	ndarray of shape 1-by-m. This ndarray is the gradient of the quantity of
	interest from the simulation. Often, the function is a wrapper to a larger
	simulation code.

	NB JMH: This function is provided for compatability.  The output processing
	of SimulationRunner has been upgraded such that can handle gradient outputs
	just as well.  So, we've simply subclassed for now.
	"""
	from warnings import warn
	warn("this code is depricated")
	return SimulationRunner(*args, **kwargs)
