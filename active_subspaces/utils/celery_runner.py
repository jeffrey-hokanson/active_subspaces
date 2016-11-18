#import marshal, types
import dill
from celery import Celery
celery = Celery('active-subspace')
celery.config_from_object('celeryconfig')

# http://stackoverflow.com/questions/1253528/is-there-an-easy-way-to-pickle-a-python-function-or-otherwise-serialize-its-cod/1253813#1253813

@celery.task()
def celery_runner(x, marshal_func, args, kwargs):
	func = dill.loads(marshal_func)
	# Marshal approach
	#code = marshal.loads(marshal_func)
	#func = types.FunctionType(code, globals(), "func")
	return func(x, *args, **kwargs)

