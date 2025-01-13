#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --partition=soc-kp
#SBATCH --account=soc-kp
#SBATCH -N 1 
#SBATCH -J CG_ex5-serial-sim
#SBATCH -o CG_ex5-serial-sim.log
#SBATCH -e CG_ex5-serial-sim.log
#set CC=/uufs/chpc.utah.edu/sys/installdir/gcc/5.4.0/bin/gcc
#set CXX=/uufs/chpc.utah.edu/sys/installdir/dcc/5.4.0/bin/g++
source /uufs/chpc.utah.edu/common/home/u1260704/Tools/Seesaw/Seesaw_CHPC.env


# noAbs
python3 ../../../../src/satire+.py \
        --std \
		--file CG_ex_5_K20_N27.txt \
		--logfile CG_ex_5_K20_N27.txt_abs-serial_noAbs.pylog \
		--outfile CG_ex_5_K20_N27.txt_abs-serial_noAbs.out