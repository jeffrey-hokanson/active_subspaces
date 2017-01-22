#! /opt/python/gcc/2.7.11/bin/python

from active_subspaces import celery
celery.control.broadcast('shutdown')
