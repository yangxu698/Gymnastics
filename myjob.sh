#!/bin/csh
 #$ -M yxu6@nd.edu     # Email address for job notification
 #$ -m abe               # Send mail when job begins, ends and aborts
 #$ -q long              # Specify queue
 #$ -pe smp 1            # Specify number of cores to use.
 #$ -N pypanda        # Specify job name

 module load python/3.6.4

 python3 pypanda.py
