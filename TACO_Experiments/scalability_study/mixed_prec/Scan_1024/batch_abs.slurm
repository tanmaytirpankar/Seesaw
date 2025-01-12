#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --partition=soc-kp
#SBATCH --account=soc-kp
#SBATCH -N 1 
#SBATCH -J Scan_1024pt_sat_abs-serial-sim
#SBATCH -o Scan_1024pt_sat-abs-serial-sim.log
#SBATCH -e Scan_1024pt_sat-abs-serial-sim.log
#set CC=/uufs/chpc.utah.edu/sys/installdir/gcc/5.4.0/bin/gcc
#set CXX=/uufs/chpc.utah.edu/sys/installdir/dcc/5.4.0/bin/g++
source /uufs/chpc.utah.edu/common/home/u1260704/Tools/Seesaw/Seesaw_CHPC.env

# No-Abs window 
python3 ../../../../src/satire+.py \
        --std \
		--file Scan_1024pt_sat.txt \
		--logfile Scan_1024pt_sat_abs-serial_noAbs.pylog \
		--outfile Scan_1024pt_sat_abs-serial_noAbs.out
