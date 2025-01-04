#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --partition=soc-kp
#SBATCH --account=soc-kp
#SBATCH -N 1 
#SBATCH -J matmul_128_128_128_abs-serial-sim
#SBATCH -o matmul_128_128_128-abs-serial-sim.log
#SBATCH -e matmul_128_128_128-abs-serial-sim.log
#set CC=/uufs/chpc.utah.edu/sys/installdir/gcc/5.4.0/bin/gcc
#set CXX=/uufs/chpc.utah.edu/sys/installdir/dcc/5.4.0/bin/g++
source /uufs/chpc.utah.edu/common/home/u1260704/Tools/Seesaw/Seesaw_CHPC.env


# No-Abs window
python3 ../../src/satire.py \
        --std \
		--compress \
		--file matmul_128_128_128.txt \
		--logfile matmul_128_128_128_abs-serial_noAbs.pylog \
		--outfile matmul_128_128_128_abs-serial_noAbs.out