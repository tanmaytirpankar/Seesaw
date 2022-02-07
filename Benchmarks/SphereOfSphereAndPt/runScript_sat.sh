
testname="EigenSphere.txt"

# Different run configurations
#1. Confiig : Basic Solver + instability
python3 ../../src/seesaw.py  \
        --std \
        --enable-abstraction --mindepth 10 --maxdepth 20 \
		--file ${testname} \
		--logfile basic.log \
		--outfile basic.out

#2. Confiig : Basic Solver + instability + realpaver
python3 ../../src/seesaw.py  \
        --std \
        --enable-abstraction --mindepth 10 --maxdepth 20 \
		--file ${testname} \
		--realpaver
		--logfile basic_RP.log \
		--outfile basic_RP.out


#3. Confiig : Basic Solver + instability
python3 ../../src/seesaw.py  \
        --std \
        --enable-abstraction --mindepth 10 --maxdepth 20 \
        --report-instability \
		--file ${testname} \
		--logfile basic_instab.log \
		--outfile basic_instab.out


#4. Confiig : Basic Solver + instability + realpaver
python3 ../../src/seesaw.py  \
        --std \
        --enable-abstraction --mindepth 10 --maxdepth 20 \
        --report-instability \
		--file ${testname} \
		--realpaver
		--logfile basic_instab_RP.log \
		--outfile basic_instab_RP.out

#5. Config : Basic Solver + statistical error with fractional threshold
python3 ../../src/seesaw.py  \
        --std \
        --enable-abstraction --mindepth 10 --maxdepth 20 \
        --report-instability \
		--file ${testname} \
		--stat-err-enable \
		--stat-err-fraction 0.5 \
		--logfile basic_instab_stat_0.5.log \
		--outfile basic_instab_stat_0.5.out

#6. Config : Basic Solver + realpaver + statistical error with fractional threshold
python3 ../../src/seesaw.py  \
        --std \
        --enable-abstraction --mindepth 10 --maxdepth 20 \
        --report-instability \
		--file ${testname} \
		--realpaver \
		--stat-err-enable \
		--stat-err-fraction 0.5 \
		--logfile basic_instab_RP_stat_0.5.log \
		--outfile basic_instab_RP_stat_0.5.out

## ---------------- Constrained optimizer enabled -------------------------

#6. Confiig : Constrained Solver + instability
python3 ../../src/seesaw.py  \
        --std \
        --enable-abstraction --mindepth 10 --maxdepth 20 \
		--file ${testname} \
		--enable-constr \
		--logfile Constrained.log \
		--outfile Constrained.out

#7. Confiig : Constrained Solver + instability + realpaver
python3 ../../src/seesaw.py  \
        --std \
        --enable-abstraction --mindepth 10 --maxdepth 20 \
		--file ${testname} \
		--enable-constr \
		--realpaver
		--logfile Constrained_RP.log \
		--outfile Constrained_RP.out


#8. Confiig : Constrained Solver + instability
python3 ../../src/seesaw.py  \
        --std \
        --enable-abstraction --mindepth 10 --maxdepth 20 \
        --report-instability \
		--file ${testname} \
		--enable-constr \
		--logfile Constrained_instab.log \
		--outfile Constrained_instab.out


#9. Confiig : Constrained Solver + instability + realpaver
python3 ../../src/seesaw.py  \
        --std \
        --enable-abstraction --mindepth 10 --maxdepth 20 \
        --report-instability \
		--file ${testname} \
		--enable-constr \
		--realpaver
		--logfile Constrained_instab_RP.log \
		--outfile Constrained_instab_RP.out

