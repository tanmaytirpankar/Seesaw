

import sys
import glob

fout = open("Results.txt", 'w')

configs = ['noAbs']
Message = {'noAbs' : 'Without Abstraction ', \
		   }

BenchmarkNames = {'advect3d' : 'Advection3D', \
				  'fdtd1d_t64' : 'FDTD', \
				  'matmul64' : 'Matrix Multiplication - 64x64', \
				  'matmul128' : 'Matrix Multiplication - 128x128', \
				  'FFT_1024' : 'FFT_1024', \
				  'FFT_4096pt' : 'FFT_4096', \
				  'Scan_1024' : 'Scan_1024(Prefix sum)', \
				  'Scan_4096' : 'Scan_4096(Prefix Sum)', \
				  'CG_Arc' : 'Conjugate gradient (ARC130)', \
				  'CG_Pores' : 'Conjugate gradient (Pores)', \
				  'chainSum' : 'Chain Sum', \
				  'horner' : 'Horner', \
				  'polyEval' : 'Polynomial Evaluation', \
				  'reduction' : 'Reduction', \
				  }

test_name = sys.argv[1]
print(test_name)
fout.write(test_name+"\n")
file_list = list(glob.iglob('*'))

pylog_dict = dict()
outlog_dict = dict()
#print(file_list)
pylog_dict['noAbs'] = list(filter( lambda x: 'pylog' in x and 'noAbs' in x, file_list))

outlog_dict['noAbs'] = list(filter(lambda x: 'out' in x and 'noAbs' in x , file_list))
#print(pylog_dict)
BenchName = BenchmarkNames.get(test_name, test_name)
print("****** Benchmark :", BenchName, "**************")
fout.write("****** Benchmark : {bench} **************\n".format(bench=BenchName))
for conf in configs:
	if (len(pylog_dict[conf]) == 0):
		print(Message[conf], "did not execute \n")
		fout.write("{message} -- did not execute \n".format(message=Message[conf]))
	else:
		pylogname = pylog_dict[conf][0]
		outlogname = outlog_dict[conf][0]
		#print(pylogname, outlogname)
		logfile = open(pylogname, 'r').read().splitlines()
		outfile = open(outlogname, 'r').read().splitlines()
		AST_DEPTH = list(filter(lambda x: "AST_DEPTH" in x, logfile))
		ABSOLUTE_ERROR = list(filter(lambda x: "ABSOLUTE_ERROR" in x, outfile))
		EXECUTION_TIME = list(filter(lambda x: "Full time" in x, outfile))

		error = max(list(map(lambda x : float(x.split(':')[1]), ABSOLUTE_ERROR)))

		print("\t",Message[conf], "->", "execution time :", EXECUTION_TIME[0])
		print("\t",Message[conf], "->", "absolute error :", error)
		fout.write("\t {message} -->  execution time = {exec_time}\n".format(message=Message[conf], exec_time=EXECUTION_TIME[0]))
		fout.write("\t {message} -->  absolute error = {abs_err}\n\n\n".format(message=Message[conf], abs_err=error))


fout.close()
