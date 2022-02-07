testname="interpolator.txt"


for frac in 0.0 0.5 0.8 
do

	python3 ../../../src/seesaw.py \
			--std \
			--file ${testname} \
			--stat-err-enable \
			--stat-err-fraction ${frac} \
			--samples 10000 \
			--logfile stat_basic_F${frac}.log \
			--outfile stat_basic_F${frac}.out 

	python3 ../../../src/seesaw.py \
			--std \
			--file ${testname} \
			--realpaver \
			--stat-err-enable \
			--stat-err-fraction ${frac} \
			--samples 10000 \
			--logfile stat_basic_RP_F${frac}.log \
			--outfile stat_basic_RP_F${frac}.out 
done
