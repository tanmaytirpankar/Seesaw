#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --partition=soc-kp
#SBATCH --account=soc-kp
#SBATCH -N 1 
#SBATCH -J fdtd1d_64_abs-serial-sim
#SBATCH -o fdtd1d_64-abs-serial-sim.log
#SBATCH -e fdtd1d_64-abs-serial-sim.log
#set CC=/uufs/chpc.utah.edu/sys/installdir/gcc/5.4.0/bin/gcc
#set CXX=/uufs/chpc.utah.edu/sys/installdir/dcc/5.4.0/bin/g++
source /uufs/chpc.utah.edu/common/home/u1260704/Tools/Seesaw/Seesaw_CHPC.env


# no-Abs 
python3 ../../../src/satire+.py \
        --std \
		--compress \
		--file fdtd1d_64.txt \
		--logfile fdtd1d_64_abs-serial_noAbs.pylog \
		--outfile fdtd1d_64_abs-serial_noAbs.out
