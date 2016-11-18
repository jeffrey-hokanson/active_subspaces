#! /bin/bash
# Note: to run on the a cluster with slurm, run:
# > sbatch start_celery.sh

##SBATCH -o outz-%j
##SBATCH -e errz-%j
#SBATCH --job-name="celery"
#SBATCH --uid=hokanson
##SBATCH -p pconstan
#SBATCH --nodes=1	## The max is 2 for Paul's cluster
#SBATCH --ntasks-per-node=1 	## the max is 16 for Paul's cluster
#SBATCH --share

# Note, you will need to run this script in the active_subspaces root 
# directory so that it can pull from the right celeryconfig
cd $HOME/active_subspaces
export TMPDIR=$SCRATCH
#srun celery -A active_subspaces.celery worker -n worker-$SLURM_ARRAY_TASK_ID
#srun celery -A active_subspaces.celery worker -n worker-$SLURM_JOB_ID --concurrency=1
#mpirun -np 1 -wdir $HOME/active_subspaces celery -A active_subspaces.celery worker -n worker-$SLURM_JOB_ID  --concurrency=1


# These options force MPI to ignore the psm and openib layer (part of the Intel Infinaband)
# The issue is that the Infiniband only allows a finite number of endpoints, which we exhaust
# However, eventually we may want each worker to have access to multiple compute nodes.
# See the documentation below for help.
# http://doc.escience-lab.org/elcid/elcid-Job-submission.html
mpirun -np 1 -wdir $HOME/active_subspaces --mca mtl ^psm --mca btl ^openib  celery -A active_subspaces.celery worker -n worker-$SLURM_JOB_ID  --concurrency=1
