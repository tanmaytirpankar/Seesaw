import sys
import copy

if __name__ == "__main__":
	print("1: Matrix file name")
	print("2: N size")
	print("3: tagName")
	print("4: in error value")

	# L is the number of iterations
	# k is the iteration number
	print("5: num iterations")
	filename = sys.argv[1]
	N = int(sys.argv[2])
	tagname = sys.argv[3]
	err = float(sys.argv[4])

	L = int(sys.argv[5])


	## Read-in the A matrix
	A = [[0]*N for i in range(N)]
	b = [0]*N
	lines = open(filename, 'r').read().splitlines()
	for i in range(0,N*N):
		line = lines[i]
		[row,col,val] = line.split(':')
		A[int(row)][int(col)] = float(val)
	for i in range(N*N, N*N+N):
		line = lines[i]
		[row, val] = line.split(':')
		b[int(row)] = float(val)



	dumpStr = ""
	k = 0
	R = [[0]*N for i in range(2)]
	P = [[0]*N for i in range(2)]

	## Compute the first residual by r0 = b - A*x0
	for i in range(0, N):
		rhs_accumulated_matvec_i = 0
		for j in range(N):
			lhs_matvec_i_elem_j = "Ax_{idx}_{jdy}".format(idx=str(i), jdy=j)
			rhs_matvec_i_elem_j = "A_{idx}_{jdy}*x_{kid}_{jdy}".format(kid=str(k), idx=str(i), jdy=j)
			dumpStr += lhs_matvec_i_elem_j + " rnd64 = " + rhs_matvec_i_elem_j + ";\n"

			rhs_accumulated_matvec_i = lhs_matvec_i_elem_j if j == 0 else lhs_accumulated_matvec_i + "+" + lhs_matvec_i_elem_j
			lhs_accumulated_matvec_i = "Ax_acc_{idx}_{jdy}".format(idx=str(i), jdy=j)
			dumpStr += lhs_accumulated_matvec_i + " rnd64 = " + rhs_accumulated_matvec_i + ";\n"

		# rhs_matvec_i = "b_{idx}".format(idx=i) + "-" + "-".join(["A_{idx}_{jdy}*x_{kid}_{jdy}".format(kid=str(k), idx=str(i), jdy=j) for j in range(N)])
		rhs_matvec_i = "b_{idx}".format(idx=i) + "-" + lhs_accumulated_matvec_i
		lhs_matvec_i = "r_{kid}_{idx}".format(kid=str(k), idx=i)
		R[0][i] = lhs_matvec_i
		dumpStr += lhs_matvec_i + " rnd64 = " + rhs_matvec_i + ";\n"
	P = copy.deepcopy(R)

	while(k < L):
		## alpha = r^T r
		rhs_accumulated_RTR_i = 0
		for i in range(0, N):
			rhs_accumulated_RTR_i = R[0][i] + "*" + R[0][i] if i == 0 else lhs_accumulated_RTR_i + "+" + R[0][i] + "*" + \
																		   R[0][i]
			lhs_accumulated_RTR_i = "rtr_acc_{idx}".format(idx=i)
			if i != N-1:
				dumpStr += lhs_accumulated_RTR_i + " rnd32 = " + rhs_accumulated_RTR_i + ";\n"

		# rhs_RTR = "+".join(["{Ri}*{Ri}".format(Ri = R[0][i]) for i in range(N)])
		rhs_RTR = rhs_accumulated_RTR_i
		lhs_RTR = "rtr_{kid}".format(kid=k)
		dumpStr += lhs_RTR + " rnd32 = " + rhs_RTR + ";\n"

		rhs_AP = [0]*N

		for i in range(0, N):
			rhs_accumulated_AP_i = 0
			for j in range(N):
				rhs_accumulated_AP_i = "A_{idx}_{jdy} * {Pi}".format(idx = str(i), jdy = str(j), Pi = P[0][i]) if j == 0 else lhs_accumulated_AP_i + "+" + "A_{idx}_{jdy} * {Pi}".format(idx = str(i), jdy = str(j), Pi = P[0][i])
				lhs_accumulated_AP_i = "AP_acc_{idx}_{jdy}".format(idx=str(i), jdy=j)
				if j != N-1:
					dumpStr += lhs_accumulated_AP_i + " rnd32 = " + rhs_accumulated_AP_i + ";\n"

			# rhs_AP_i = "+".join(["A_{idx}_{jdy} * {Pi}".format(idx = str(i), jdy = str(j), Pi = P[0][i]) for j in range(N)])
			rhs_AP_i = rhs_accumulated_AP_i
			lhs_AP_i = "AP_{kid}_{idx}".format(kid=str(k), idx=i)

			rhs_AP[i] = lhs_AP_i
			dumpStr += lhs_AP_i + " rnd32 = " + rhs_AP_i + ";\n"

		rhs_accumulated_PAP = 0
		for i in range(N):
			rhs_accumulated_PAP = P[0][i] + "*" + rhs_AP[i] if i == 0 else lhs_accumulated_PAP + "+" + P[0][i] + "*" + \
																		   rhs_AP[i]
			lhs_accumulated_PAP = "PAP_acc_{idx}".format(idx=i)
			if i != N-1:
				dumpStr += lhs_accumulated_PAP + " rnd32 = " + rhs_accumulated_PAP + ";\n"
		# rhs_PAP = "+".join(["{Pi}*{APi}".format(Pi = P[0][i], APi = rhs_AP[i]) for i in range(N)])
		rhs_PAP = rhs_accumulated_PAP
		lhs_PAP = "pap_{kid}".format(kid=k)
		dumpStr += lhs_PAP + " rnd32 = " + rhs_PAP + ";\n"

		rhs_alpha = "({numer})/({denom})".format(numer = lhs_RTR, denom = lhs_PAP)
		lhs_alpha = "alpha_{kid}".format(kid = k)
		dumpStr += lhs_alpha + " rnd32 = " + rhs_alpha + ";\n"


		## update x_{k+1} = x_k + alpha*r_k
		for i in range(N):
			lhs_alpha_rid = "alpha_rid_{idx}".format(idx=i)
			rhs_alpha_rid = lhs_alpha + "*" + R[0][i]
			dumpStr += lhs_alpha_rid + " rnd64 = " + rhs_alpha_rid + ";\n"
			# rhs_x = "x_{kid}_{idx} + {alpha}*{Rid}".format(alpha = lhs_alpha, kid = k, idx = i, Rid = R[0][i])
			rhs_x = "x_{kid}_{idx} + {alpha_rid}".format(alpha_rid = lhs_alpha_rid, kid = k, idx = i)
			lhs_x = "x_{kpid}_{idx}".format(kpid = str(k+1), idx = str(i))
			dumpStr += lhs_x + " rnd64 = " + rhs_x + ";\n"

		## update next residue r1 = r0 - alpha*AP

		for i in range(0, N):
			lhs_alpha_api = "alpha_api_{idx}".format(idx=i)
			rhs_alpha_api = lhs_alpha + "*" + rhs_AP[i]
			dumpStr += lhs_alpha_api + " rnd64 = " + rhs_alpha_api + ";\n"
			# rhs_next_residue = "{Rid} - {alpha}*{APi}".format(Rid=R[0][i], alpha=lhs_alpha, APi = rhs_AP[i])
			rhs_next_residue = "{Rid} - {alpha_api}".format(Rid=R[0][i], alpha_api=lhs_alpha_api)
			lhs_next_residue = "r_{kpid}_{idx}".format(kpid = k+1, idx= i)
			R[1][i] = lhs_next_residue

			dumpStr += lhs_next_residue + " rnd64 = " + rhs_next_residue + ";\n"

		## Compute beta
		rhs_accumulated_vecvec = 0
		for i in range(N):
			rhs_accumulated_vecvec = R[1][i] + "*" + R[1][i] if i == 0 else lhs_accumulated_vecvec + "+" + R[1][
				i] + "*" + R[1][i]
			lhs_accumulated_vecvec = "vecvec_acc_{idx}".format(idx=i)
			if i != N-1:
				dumpStr += lhs_accumulated_vecvec + " rnd32 = " + rhs_accumulated_vecvec + ";\n"
		# rhs_r1tr1 = "+".join(["{R1id}*{R1id}".format(R1id = R[1][i]) for i in range(N)])
		rhs_r1tr1 = rhs_accumulated_vecvec
		lhs_r1tr1 = "r1tr1_{kid}".format(kid=k)

		dumpStr += lhs_r1tr1 + " rnd32 = " + rhs_r1tr1 + ";\n"

		rhs_accumulated_vecvec = 0
		for i in range(N):
			rhs_accumulated_vecvec = R[0][i] + "*" + R[0][i] if i == 0 else lhs_accumulated_vecvec + "+" + R[0][
				i] + "*" + R[0][i]
			lhs_accumulated_vecvec = "vecvec_acc_{idx}".format(idx=i)
			if i != N-1:
				dumpStr += lhs_accumulated_vecvec + " rnd32 = " + rhs_accumulated_vecvec + ";\n"
		# rhs_r0tr0 = "+".join(["{R0id}*{R0id}".format(R0id = R[0][i]) for i in range(N)])
		rhs_r0tr0 = rhs_accumulated_vecvec
		lhs_r0tr0 = "r0tr0_{kid}".format(kid=k)
		dumpStr += lhs_r0tr0 + " rnd32 = " + rhs_r0tr0 + ";\n"

		rhs_beta = "({numer})/({denom})".format(numer=lhs_r1tr1, denom=lhs_r0tr0)
		lhs_beta = "beta_{kid}".format(kid=k)
		dumpStr += lhs_beta + " rnd32 = " + rhs_beta + ";\n"

		## Update P
		for i in range(0, N):
			lhs_beta_pid = "beta_pid_{idx}".format(idx=i)
			rhs_beta_pid = lhs_beta + "*" + P[0][i]
			dumpStr += lhs_beta_pid + " rnd32 = " + rhs_beta_pid + ";\n"
			# rhs_next_p = "{R1id} + {beta}*{p0id}".format(R1id = R[1][i], beta=lhs_beta, p0id=P[0][i])
			rhs_next_p = "{R1id} + {beta_pid}".format(R1id = R[1][i], beta_pid=lhs_beta_pid)
			lhs_next_p = "p_{kpid}_{idx}".format(kpid = k+1, idx=i)
			P[1][i] = lhs_next_p
			dumpStr += lhs_next_p + " rnd32 = " + rhs_next_p + ";\n"

		P[0] = P[1]
		R[0] = R[1]

		k += 1

	outfile = "CG_"+tagname+"_K"+str(k)+"_N"+str(N)+".txt"
	fout = open(outfile, 'w')

	## Print the inputs ##
	fout.write("INPUTS {\n")
	for i in range(N):
		for j in range(N):
			val = A[i][j]
			var = "A_{idx}_{jdy}".format(idx=i, jdy=j)
			if val == 0:
				fout.write("\t"+var+"\t fl64 : ("+str(val)+" , "+str(val)+");\n")
			else:
				fout.write("\t"+var+"\t fl64 : ("+str(val-err)+" , "+str(val+err)+");\n")
		bval = str(b[i])
		bvar = "b_{idx}".format(idx=i)
		fout.write("\t"+bvar+"\t fl64 : ("+bval+" , "+bval+");\n")
		bvar = "x_0_{idx}".format(idx=i)
		fout.write("\t"+bvar+"\t fl64 : (0.0 , 0.0);\n")
			
	fout.write("}\n\n")


	fout.write("OUTPUTS {\n\n")
	# fout.write("x_"+str(k)+"_"+str(N-1)+";\n")
	for i in range(k):
		fout.write("alpha_" + str(i) + ";\n")
		fout.write("beta_" + str(i)  + ";\n")
	fout.write("}\n\n")

## print the matvec operation between A and x

	fout.write("EXPRS {\n\n")
	fout.write(dumpStr)
	fout.write("}\n\n")

	fout.close()

	print(dumpStr)


