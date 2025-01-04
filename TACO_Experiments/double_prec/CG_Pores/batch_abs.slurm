#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --partition=soc-kp
#SBATCH --account=soc-kp
#SBATCH -N 1 
#SBATCH -J CG_Pores-abs-serial-sim
#SBATCH -o CG-Pores-abs-serial-sim.log
#SBATCH -e CG-Pores-abs-serial-sim.log
#set CC=/uufs/chpc.utah.edu/sys/installdir/gcc/5.4.0/bin/gcc
#set CXX=/uufs/chpc.utah.edu/sys/installdir/dcc/5.4.0/bin/g++
source /uufs/chpc.utah.edu/common/home/u1260704/Tools/Seesaw/Seesaw_CHPC.env



# ino Abs
python3 ../../../src/satire.py \
        --std \
		--file CG_Pores_noError_K20_N30_E0.0.txt \
		--logfile CG_Pores_noError_K20_N30_E0.0_abs-serial_noAbs.pylog \
		--outfile CG_Pores_noError_K20_N30_E0.0_abs-serial_noAbs.out