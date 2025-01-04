#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --partition=soc-kp
#SBATCH --account=soc-kp
#SBATCH -N 1 
#SBATCH -J poly-eval_sat_abs-serial-sim
#SBATCH -o poly-eval_sat-abs-serial-sim.log
#SBATCH -e poly-eval_sat-abs-serial-sim.log
#set CC=/uufs/chpc.utah.edu/sys/installdir/gcc/5.4.0/bin/gcc
#set CXX=/uufs/chpc.utah.edu/sys/installdir/dcc/5.4.0/bin/g++
source /uufs/chpc.utah.edu/common/home/u1260704/Tools/Seesaw/Seesaw_CHPC.env


# No-Abs window 
python3 ../../../src/satire+.py \
        --std \
		--file f_polynomial_50.txt \
		--logfile poly-eval_sat_abs-serial_noAbs.pylog \
		--outfile poly-eval_sat_abs-serial_noAbs.out 
