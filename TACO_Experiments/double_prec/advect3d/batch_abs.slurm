#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --partition=soc-kp
#SBATCH --account=soc-kp
#SBATCH -N 1 
#SBATCH -J advect3d_20_abs-serial-sim
#SBATCH -o advect3d_20-abs-serial-sim.log
#SBATCH -e advect3d_20-abs-serial-sim.log
#set CC=/uufs/chpc.utah.edu/sys/installdir/gcc/5.4.0/bin/gcc
#set CXX=/uufs/chpc.utah.edu/sys/installdir/dcc/5.4.0/bin/g++
source /uufs/chpc.utah.edu/common/home/u1260704/Tools/Seesaw/Seesaw_CHPC.env


# no-abs
python3 ../../../src/satire.py \
        --std \
		--file advect3d_20.txt \
		--logfile advect3d_20_abs-serial_noAbs.pylog \
		--outfile advect3d_20_abs-serial_noAbs.out