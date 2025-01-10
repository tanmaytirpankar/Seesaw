import sys
import random
import copy


def polynom(N, B, fout):

	A = copy.deepcopy(B)

	fout.write("INPUTS {\n\n")
	fout.write("\t x fl32 : (-1.0, 1.0) ;\n")
	fout.write("}\n")

	Acc = []

	fout.write("OUTPUTS {\n\n")
	fout.write("\t Final ;\n")
	fout.write("}\n")

	fout.write("EXPRS {\n\n")
	
	for n in range(N,0,-1):

		for i in range(n):
			x_lhs = "x_{i}".format(i=i)
			x_rhs = "x" if i==0 else "{x_prev}*x".format(x_prev="x_"+str(i-1))
			fout.write("{x_lhs} rnd32 = {x_rhs} ;\n".format(x_lhs=x_lhs, x_rhs=x_rhs))
		
		lhs = "S_{ND}".format(ND=n)
		# rhs = "({coeff}*{polyn})".format(coeff = A[n], polyn = "( "+" * ".join(["x" for i in range(n)])+" )")
		rhs = "({coeff}*{polyn})".format(coeff = A[n], polyn = "x_"+str(n-1))
		fout.write("{lhs} rnd32 = {rhs} ;\n".format(lhs=lhs, rhs=rhs))
		Acc.append(lhs)

	fout.write("Final rnd32 = {a0} + {acc} ;\n".format(a0=A[0], acc="+".join(Acc)))

	fout.write("}\n")



def horner(N, B, fout):

	A = copy.deepcopy(B)

	fout.write("INPUTS {\n\n")
	fout.write("\t x fl32 : (-1.0, 1.0) ;\n")
	fout.write("}\n")
	

	fout.write("OUTPUTS {\n\n")
	fout.write("\t S_{ND} ;\n".format(ND=0))
	fout.write("}\n")

	fout.write("EXPRS {\n\n")
	
	for n in range(N, 0,-1):
		
		print(n, A)

		prod_lhs = "prod_{ND}".format(ND=n)
		prod_rhs = "(x*{a_n})".format(a_n=A[n])

		fout.write("{prod_lhs} rnd32 = {prod_rhs} ;\n".format(prod_lhs=prod_lhs, prod_rhs=prod_rhs))

		lhs = "S_{ND}".format(ND=n-1)
		# rhs = "({a_nm1} + x*{a_n})".format(a_nm1 = A[n-1], a_n = A[n])

		rhs = "({a_nm1} + {prod})".format(a_nm1 = A[n-1], prod=prod_lhs)


		fout.write("{lhs} rnd32 = {rhs} ;\n".format(lhs=lhs, rhs=rhs))
		A[n-1] = lhs

	fout.write("}\n")




if __name__ == "__main__":
	
	N = int(sys.argv[1])

	fpoly = open("f_polynomial_"+str(N)+".txt", 'w')
	fhorner = open("f_horner_"+str(N)+".txt", 'w')

	B = [random.uniform(-10.0, 10.0) for i in range(N+1)]
	print(len(B))

	horner(N, B, fhorner)
	polynom(N, B, fpoly)

	fpoly.close()
	fhorner.close()
