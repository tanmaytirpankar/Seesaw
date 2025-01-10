
import sys


def fdtd1d(E, H, N, T, fout):

	for t in range(1,T+1):
		
		for i in range(1,N):
			h_diff_rhs = "({h_i} - {h_im1});".format( \
						h_i = H[i], \
						h_im1 = H[i-1] \
					  )
			h_diff_lhs = "{h_diff_i}".format(h_diff_i = "H_diff_"+str(i)+"_"+str(t))

			fout.write(h_diff_lhs + " rnd32 = " + h_diff_rhs + "\n")
			
			h_diff_factor_rhs = "(0.5*{h_diff_i});".format(h_diff_i = h_diff_lhs)
			h_diff_factor_lhs = "{h_diff_factor_i}".format(h_diff_factor_i = "H_diff_factor_"+str(i)+"_"+str(t))

			fout.write(h_diff_factor_lhs + " rnd32 = " + h_diff_factor_rhs + "\n")

			e_lhs = "{e_i_t}".format(e_i_t="E_" + str(i) + "_" + str(t))

			# e_rhs = "({e_i} - 0.5*({h_i} - {h_im1}));".format( \
			# 	e_i=E[i], \
			# 	h_i=H[i], \
			# 	h_im1=H[i - 1] \
			# 	)
			
			e_rhs = "({e_i} - {h_diff_factor});".format( \
				e_i=E[i], \
				h_diff_factor=h_diff_factor_lhs \
				)

			E[i] = e_lhs

			fout.write(e_lhs + " rnd32 = " + e_rhs + "\n")

		for i in range(0,N):
			e_diff_rhs = "({e_ip1} - {e_i});".format( \
						e_ip1 = E[i+1], \
						e_i = E[i] \
					  )
			e_diff_lhs = "{e_diff_i}".format(e_diff_i = "E_diff_"+str(i)+"_"+str(t))

			fout.write(e_diff_lhs + " rnd32 = " + e_diff_rhs + "\n")
			
			e_diff_factor_rhs = "(0.7*{e_diff_i});".format(e_diff_i = e_diff_lhs)
			e_diff_factor_lhs = "{e_diff_factor_i}".format(e_diff_factor_i = "E_diff_factor_"+str(i)+"_"+str(t))

			fout.write(e_diff_factor_lhs + " rnd32 = " + e_diff_factor_rhs + "\n")
			
			h_lhs = "{h_i_t}".format(h_i_t = "H_"+str(i)+"_"+str(t))

			# h_rhs = "({h_i} - 0.7*({e_ip1} - {e_i}));".format( \
			# 			h_i = H[i], \
			# 			e_ip1 = E[i+1], \
			# 			e_i = E[i] \
			# 		 )
			
			h_rhs = "({h_i} - {e_diff_factor});".format( \
				h_i=H[i], \
				e_diff_factor=e_diff_factor_lhs \
				)

			H[i] = h_lhs

			fout.write(h_lhs + " rnd32 = " + h_rhs + "\n")


if __name__ == "__main__":

	N = int(sys.argv[1])
	T = int(sys.argv[2])

	E = ['E_'+str(i)+'_0' for i in range(0,N+1)]
	H = ['H_'+str(i)+'_0' for i in range(0,N)]

	fout = open("fdtd1d_"+str(N)+"_"+str(T)+".txt",'w')

	fout.write("INPUTS {\n")

	block_cnt = 0
	for i in range(0, N+1):
		
		lb = '0.0' #str(float(i-block_cnt)/N)
		ub = '1.0' #str(float(i-block_cnt+20)/N)
		block_cnt += 1
		
		fout.write("\t E_"+str(i)+"_0\t fl32 :  ("+lb+" , "+ub+");\n")
		if(i < N):
			fout.write("\t H_"+str(i)+"_0\t fl32:  ("+lb+" , "+ub+");\n")

		if block_cnt==5:
			block_cnt = 0

	fout.write("}\n\n")
		
	fout.write("OUTPUTS {\n")
	#fout.write("E_"+str(int(N/2))+"_"+str(T)+";\n")
	fout.write("H_"+str(int(N/2))+"_"+str(T)+";\n")
	fout.write("}\n")

	fout.write("EXPRS {\n")
	fdtd1d(E, H, N, T, fout)
	fout.write("}\n")

	fout.close()



