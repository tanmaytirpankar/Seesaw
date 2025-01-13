#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --partition=soc-kp
#SBATCH --account=soc-kp
#SBATCH -N 1 
#SBATCH -J benchmarks_abs-serial-sim
#SBATCH -o benchmarks-abs-serial-sim.log
#SBATCH -e benchmarks-abs-serial-sim.log
#set CC=/uufs/chpc.utah.edu/sys/installdir/gcc/5.4.0/bin/gcc
#set CXX=/uufs/chpc.utah.edu/sys/installdir/dcc/5.4.0/bin/g++
source /uufs/chpc.utah.edu/common/home/u1260704/Tools/Seesaw/Seesaw_CHPC.env


DIRS="CG_ex5"

set -x

echo "Here: $GPHOME"
GPHOME1="/uufs/chpc.utah.edu/common/home/u1260704/Tools/gelpia/"

# use only if $GPHOME is not defined globally else use the global env value
if [[ -z "${GPHOME}$" ]]; then

	if [[ -d $GPHOME1 ]]; then
		GPHOME=$GPHOME1
	else
		echo "GPHOME1 is set to $GPHOME1 : no such directory --  Fix GPHOME path to gelpia home directory"
	fi
else
  if [[ -d "${GPHOME}$" ]]; then
		echo "$GPHOME is a valid directory"
	 else
		if [[ -d $GPHOME1 ]]; then
			GPHOME=$GPHOME1
		else
			echo "GPHOME1 is set to $GPHOME1 : no such directory --  Fix GPHOME path to gelpia home directory"
		fi
	 	
	 fi
fi

GPEXE1=$GPHOME/bin/gelpia

GPEXE2=$(eval "which gelpia")

#EXE1 = $(eval "sed -i s/\/\//\//g $GPEXE1")
#EXE2 = $(eval "sed -i s/\/\//\//g $GPEXE2")

EXE1=$(eval "echo ${GPEXE1//\/\//\/}")
EXE2=$(eval "echo ${GPEXE2//\/\//\/}")

if [[ $EXE1 == $EXE2 ]];
then

#find $GPHOME -name "*generated*"

	if [[ -f Results.txt ]]
	then
		echo "Removing old results file"
		rm -rf Results.txt
	fi
	
	for d in $DIRS
	do
		echo $d
		cd $d
		echo "executing batch scipt for " $d
		bash batch_abs.slurm
		python3 ../collect_results.py $d
		echo $PWD
		cat Results.txt >> ../Results.txt
		echo "Cleaning up gelpia generated files"
		find $GPHOME -name "*generated*" | xargs rm -rf
		cd ..
	done
	
	echo "Open Results.txt for summary of the evaluation"
else
	echo "COrrectly set 'GPHOME' to gelpia home diretcory"
fi
exit
