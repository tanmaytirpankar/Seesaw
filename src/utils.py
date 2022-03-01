
import copy
import os
import sys
import re
import math as mt
import time
#import numpy as np
import Globals
#import sympy as sym
import random
import struct
import time
import subprocess as sb
from multiprocessing import Process, Value, Manager
import hashlib
from collections import OrderedDict
import symengine as seng
import sympy

from itertools import tee

import shutil
import os.path as path
gelpia_path = shutil.which("gelpia")
gelpia_dir = path.dirname(gelpia_path)

sys.path.append(gelpia_dir)
import gelpia
import gelpia_logging as logging
logging.set_log_level(logging.QUIET)
logging.set_log_filename(None)


gelpia.setup_requirements(gelpia.GIT_DIR)
gelpia_rust_executable = gelpia.setup_rust_env(gelpia.GIT_DIR, False)

gelpia_input_epsilon = 1e-4
gelpia_output_epsilon = 1e-4
gelpia_output_epsilon_relative = 1e-4
gelpia_dreal_epsilon = 1e-15
gelpia_dreal_epsilon_relative = 1e-15
#gelpia_epsilons = (gelpia_input_epsilon, gelpia_output_epsilon, gelpia_output_epsilon_relative)
gelpia_epsilons = (gelpia_input_epsilon, gelpia_output_epsilon, gelpia_output_epsilon_relative, gelpia_dreal_epsilon, gelpia_dreal_epsilon_relative)
gelpia_timeout = 10
gelpia_grace = 0
gelpia_update = 0
gelpia_max_iters = 20000
gelpia_seed = 0

timeout = 10000


def hashSig( inSig, alg ):
	hobj = hashlib.md5(str(inSig).encode('utf-8'))
	return hobj.hexdigest()

	
def get_inputString(inputs):
    ret_list = list()
    for name,val in inputs.items():
        ret_list += [name, " = ", str(val["INTV"]), ";"]
    return "".join(ret_list)

def split_gelpia_format(msg):
	#print(msg)
	return msg.split("{")[0]\
								 .split("]")[0]\
								 .split("[")[-1]\
								 .split(",")




#def extract_input_dep(free_syms):
#	ret_list = list()
#	flist = [str(i) for i in free_syms]
#	flist.sort()
#	print("Flist:", flist)
#	for fsyms in flist:
#		ret_list += [str(fsyms), " = ", str(Globals.inputVars[seng.var(fsyms)]["INTV"]), ";"]
#	return "".join(ret_list)

def rpVariableStr( cond_free_symbols ):
	ret_list = list()
	flist = [str(i) for i in cond_free_symbols]
	flist.sort()
	for fsyms in flist:
		ret_list += ["{Variable} in {intv}".format(Variable=fsyms, intv=str(Globals.inputVars[seng.var(fsyms)]["INTV"]))]
		#ret_list += [str(fsyms), " in ", str(Globals.inputVars[seng.var(fsyms)]["INTV"]), ";"]
	retStr = " ".join(["Variables", ", ".join(ret_list)+" ;"])
	return [retStr, len(flist)]
	

# the two inputs provided are in compatible forms only requiring to be Anded together
# hence process them together before sending pre-processing for RealPaver
def process_conditionals( innerConds, externConds ):

	str_inner = str(innerConds)
	str_outer = str(externConds)
	str_cond_expr = " & ".join([str_inner]+([] if externConds is None or len(str_outer)==0 else [str_outer]))
	str_cond_expr = re.sub(r'\&\&', "&", str_cond_expr)
	str_cond_expr = re.sub(r'\|\|', "|", str_cond_expr)
	str_cond_expr = re.sub(r'\&', "&&", str_cond_expr)
	str_cond_expr = re.sub(r'\|', "||", str_cond_expr)
	str_cond_expr = re.sub(r'\*\*', "^", str_cond_expr)
	str_cond_expr = re.sub(r'Abs', "abs", str_cond_expr)
	str_cond_expr = re.sub(r're\b', "", str_cond_expr)
	str_cond_expr = re.sub(r'im\b', "0.0*", str_cond_expr)
	str_cond_expr = re.sub(r'\<\<True\>\>', "<<(True)>>", str_cond_expr)
	str_cond_expr = re.sub(r'\<\<False\>\>', "<<(False)>>", str_cond_expr)
	#str_cond_expr = re.sub(r'\<\<', "(", str_cond_expr)
	#str_cond_expr = re.sub(r'\>\>', ")", str_cond_expr)

	return str_cond_expr

# Processes command string to invoke gelpia and returns the result.
def invoke_gelpia(symExpr, cond_expr, externConstraints, inputStr, label="Func-> Dur:"):
	# print("Invoking Gelpia")
	# print("Symbolic Expression")
	# print(symExpr)
	# print("Conditional Expression")
	# print(cond_expr)
	# print("External Constraints")
	# print(externConstraints)
	# print("Input String")
	# print(inputStr)
	# exit(0)
	#try:
	#    const_intv = float(str(symExpr))
	#    return [const_intv, const_intv]
	#except ValueError:
	#    pass
	
	#print("In gelpia", seng.count_ops(symExpr))
	#print(seng.count_ops(symExpr))
	str_expr = re.sub(r'\*\*', "^", str(symExpr))
	str_expr = re.sub(r'Abs', "abs", str_expr)
	str_expr = re.sub(r're\b', "", str_expr)
	str_expr = re.sub(r'im\b', "0.0*", str_expr)

	if cond_expr == Globals.__T__ :
		str_cond_expr = "(1 <= 1)"
	else:
		str_cond_expr = str(cond_expr)
		str_cond_expr = re.sub(r'\&\&', "&", str_cond_expr)
		str_cond_expr = re.sub(r'\|\|', "|", str_cond_expr)
		str_cond_expr = re.sub(r'\&', "&&", str_cond_expr)
		str_cond_expr = re.sub(r'\|', "||", str_cond_expr)
		str_cond_expr = re.sub(r'\*\*', "^", str_cond_expr)
		str_cond_expr = re.sub(r'Abs', "abs", str_cond_expr)
		str_cond_expr = re.sub(r're\b', "", str_cond_expr)
		str_cond_expr = re.sub(r'im\b', "0.0*", str_cond_expr)
		str_cond_expr = re.sub(r'\<\<', "(", str_cond_expr)
		str_cond_expr = re.sub(r'\>\>', ")", str_cond_expr)
		str_cond_expr = re.sub(r'True', "1<=1", str_cond_expr)
		str_cond_expr = re.sub(r'False', "1<=0", str_cond_expr)

	#str_extc_expr = str(externConstraints)
	str_extc_expr = str(externConstraints)
	str_extc_expr = re.sub(r'\&\&', "&", str_extc_expr)
	str_extc_expr = re.sub(r'\|\|', "|", str_extc_expr)
	str_extc_expr = re.sub(r'\&', "&&", str_extc_expr)
	str_extc_expr = re.sub(r'\|', "||", str_extc_expr)
	str_extc_expr = re.sub(r'\*\*', "^", str_extc_expr)
	str_extc_expr = re.sub(r'Abs', "abs", str_extc_expr)
	str_extc_expr = re.sub(r're\b', "", str_extc_expr)
	str_extc_expr = re.sub(r'im\b', "0.0*", str_extc_expr)
	str_extc_expr = re.sub(r'\<\<', "(", str_extc_expr)
	str_extc_expr = re.sub(r'\>\>', ")", str_extc_expr)
	str_extc_expr = re.sub(r'True', "1<=1", str_extc_expr)
	str_extc_expr = re.sub(r'False', "1<=0", str_extc_expr)
	#print("Pass conversion gelpia")
	gstr_expr = inputStr + str_expr  ## without the constraints
	Globals.gelpiaID += 1
	#print("Constr?", Globals.enable_constr, " Begining New gelpia query->ID:", Globals.gelpiaID)
	str_constraint = " && ".join([str_cond_expr]+([] if str_extc_expr is None or len(str_extc_expr)==0 else [str_extc_expr]))
	if Globals.argList.gverbose:
		fout = open("gelpia_"+str(Globals.gelpiaID)+".txt", "w")
		fout.write("# --input-epsilon {ieps}\n".format(ieps=str(gelpia_input_epsilon)))
		fout.write("# --output-epsilon {oeps}\n".format(oeps=str(gelpia_output_epsilon)))
		fout.write("# --output-epsilon-relative {oreps}\n".format(oreps=str(gelpia_output_epsilon_relative)))
		fout.write("# --dreal-epsilon {oeps}\n".format(oeps=str(gelpia_dreal_epsilon)))
		fout.write("# --dreal-epsilon-relative {oreps}\n".format(oreps=str(gelpia_dreal_epsilon_relative)))
		fout.write("# --timeout {tout}\n".format(tout=str(gelpia_timeout)))
		fout.write("# --max-iters {miters}\n".format(miters=str(gelpia_max_iters)))
		fout.write("{x3opt}".format(x3opt="# --use-z3\n" if Globals.argList.useZ3 else ""))
		fout.write(inputStr + str_constraint +"; " + str_expr)
		fout.close()


	if Globals.enable_constr:
		gstr_expr = inputStr + str_constraint +"; " + str_expr
		print(gstr_expr)
	#fout.write(str_expr)
	##-- print(gstr_expr)

	#print(str_expr)
	start_time = time.time()
	
	max_lower = Value("d", float("nan"))
	max_upper = Value("d", float("nan"))
	manager = Manager()
	inputs_for_max = manager.dict()
	max_solver_calls = Value("i", 0)
	#print("ID:",Globals.gelpiaID, "\t Finding max, min\n")
	p = Process(target=gelpia.find_max, args=(gstr_expr,
	                                          gelpia_epsilons,
	                                          gelpia_timeout,
	                                          gelpia_grace,
	                                          gelpia_update,
	                                          gelpia_max_iters,
	                                          gelpia_seed,
	                                          False,
											  #True, #z3
											  Globals.argList.useZ3,
	                                          gelpia.SRC_DIR,
	                                          gelpia_rust_executable,
											  False, #drop constraints
	                                          max_lower,
	                                          max_upper,
											  inputs_for_max,
											  max_solver_calls))
	p.start()
	min_lower, min_upper, inputs_for_min, min_solver_calls = gelpia.find_min(gstr_expr,
	                                       gelpia_epsilons,
	                                       gelpia_timeout,
	                                       gelpia_grace,
	                                       gelpia_update,
	                                       gelpia_max_iters,
	                                       gelpia_seed,
	                                       False,
										   #True, #z3
										   Globals.argList.useZ3,
	                                       gelpia.SRC_DIR,
	                                       gelpia_rust_executable,
										   False)
	p.join()
	end_time = time.time()
	#print("Finishing gelpia query->ID:", Globals.gelpiaID)
	
	#print(str_expr)
	#print(label, end_time - start_time, "  , FSYM: ", len(symExpr.free_symbols))
	
	#return [min_lower, max_upper.value]
	#print("min_lower", min_lower, type(min_lower))
	#print("max_upper", max_upper.value, type(max_upper.value))
	total_solver_calls = min_solver_calls + max_solver_calls.value
	#print("solver_calls", total_solver_calls)
	Globals.solver_calls += total_solver_calls        
	return [min_lower if min_lower!="Overconstrained" else 0.0, \
	        max_upper.value if max_upper.value!="Overconstrained" else 0.0]


def invoke_gelpia_herror(symExpr, inputStr, label="Func-> Dur:"):
	#try:
	#    const_intv = float(str(symExpr))
	#    return [const_intv, const_intv]
	#except ValueError:
	#    pass
	
	#print("In gelpia", seng.count_ops(symExpr))
	#print(symExpr)
	str_expr = re.sub(r'\*\*', "^", str(symExpr))
	str_expr = re.sub(r'Abs', "abs", str_expr)
	str_expr = re.sub(r're\b', "", str_expr)
	str_expr = re.sub(r'im\b', "0.0*", str_expr)
	#print("Pass conversion gelpia")
	str_expr = inputStr + str_expr
	Globals.gelpiaID += 1
	#print("Begining New gelpia query->ID:", Globals.gelpiaID)
	#fout = open("gelpia_"+str(Globals.gelpiaID)+".txt", "w")
	#fout.write(str_expr)
	#fout.close()

	#print(str_expr)
	start_time = time.time()
	
	max_lower = Value("d", float("nan"))
	max_upper = Value("d", float("nan"))
	#print("ID:",Globals.gelpiaID, "\t Finding max, min\n")
	p = Process(target=gelpia.find_max, args=(str_expr,
	                                          gelpia_epsilons,
	                                          10,
	                                          gelpia_grace,
	                                          gelpia_update,
	                                          10,
	                                          gelpia_seed,
	                                          False,
	                                          gelpia.SRC_DIR,
	                                          gelpia_rust_executable,
	                                          max_lower,
	                                          max_upper))
	p.start()
	min_lower, min_upper, inputs, solver_calls = gelpia.find_min(str_expr,
	                                       gelpia_epsilons,
	                                       10,
	                                       gelpia_grace,
	                                       gelpia_update,
	                                       10,
	                                       gelpia_seed,
	                                       False,
	                                       gelpia.SRC_DIR,
	                                       gelpia_rust_executable)
	p.join()
	end_time = time.time()
	#print("Finishing gelpia query->ID:", Globals.gelpiaID)
	
	#print(str_expr)
	#print(label, end_time - start_time, "  , FSYM: ", len(symExpr.free_symbols))
	#return [min_lower, max_upper.value]
	print("min_lower", min_lower, type(min_lower))
	print("max_upper", max_upper.value, type(max_upper.value))
	return [min_lower if min_lower!="Overconstrained" else 0.0, \
	        max_upper.value if max_upper.value!="Overconstrained" else 0.0]


def extremum_of_symbolic_expression(symbolic_expression, conditional_expression, external_constraints, input_string,
									max=True):
	"""
	Sets up input for gelpia and invokes it to find extremum and corresponding input interval.

	Parameters
	----------
	symbolic_expression : TODO: Fill this
		Symbolic expression that gives the optimal value.
	conditional_expression : TODO: Fill this
		Conditional expression that acts as a constraint on the domain
	external_constraints : TODO: Fill this
		Constraints provided by the user that may narrow the domain
	input_string : string
		Input interval string
	max : bool
		Boolean value determining whether to return upper or lower bound of extrema

	Returns
	-------
	extrema_lower_bound or extrema_lower_bound : float
		Extremum value
	"""
	# Print statements for debugging
	# print("Invoking Gelpia")
	# print("Symbolic Expression")
	# print(symbolic_expression)
	# print("Conditional Expression")
	# print(conditional_expression)
	# print("External Constraints")
	# print(external_constraints)
	# print("Input String")
	# print(input_string)
	# print("Number of operations", seng.count_ops(symbolic_expression))

	# Processing symbolic expression for gelpia
	string_expression = re.sub(r'\*\*', "^", str(symbolic_expression))
	string_expression = re.sub(r'Abs', "abs", string_expression)
	string_expression = re.sub(r're\b', "", string_expression)
	string_expression = re.sub(r'im\b', "0.0*", string_expression)

	# Processing conditional expression for gelpia
	if conditional_expression == Globals.__T__:
		conditional_expression_string = "(1 <= 1)"
	else:
		conditional_expression_string = str(conditional_expression)
		conditional_expression_string = re.sub(r'\&\&', "&", conditional_expression_string)
		conditional_expression_string = re.sub(r'\|\|', "|", conditional_expression_string)
		conditional_expression_string = re.sub(r'\&', "&&", conditional_expression_string)
		conditional_expression_string = re.sub(r'\|', "||", conditional_expression_string)
		conditional_expression_string = re.sub(r'\*\*', "^", conditional_expression_string)
		conditional_expression_string = re.sub(r'Abs', "abs", conditional_expression_string)
		conditional_expression_string = re.sub(r're\b', "", conditional_expression_string)
		conditional_expression_string = re.sub(r'im\b', "0.0*", conditional_expression_string)
		conditional_expression_string = re.sub(r'\<\<', "(", conditional_expression_string)
		conditional_expression_string = re.sub(r'\>\>', ")", conditional_expression_string)
		conditional_expression_string = re.sub(r'True', "1<=1", conditional_expression_string)
		conditional_expression_string = re.sub(r'False', "1<=0", conditional_expression_string)

	# Processing external constrains for gelpia
	external_constraints_string = str(external_constraints)
	external_constraints_string = re.sub(r'\&\&', "&", external_constraints_string)
	external_constraints_string = re.sub(r'\|\|', "|", external_constraints_string)
	external_constraints_string = re.sub(r'\&', "&&", external_constraints_string)
	external_constraints_string = re.sub(r'\|', "||", external_constraints_string)
	external_constraints_string = re.sub(r'\*\*', "^", external_constraints_string)
	external_constraints_string = re.sub(r'Abs', "abs", external_constraints_string)
	external_constraints_string = re.sub(r're\b', "", external_constraints_string)
	external_constraints_string = re.sub(r'im\b', "0.0*", external_constraints_string)
	external_constraints_string = re.sub(r'\<\<', "(", external_constraints_string)
	external_constraints_string = re.sub(r'\>\>', ")", external_constraints_string)
	external_constraints_string = re.sub(r'True', "1<=1", external_constraints_string)
	external_constraints_string = re.sub(r'False', "1<=0", external_constraints_string)

	Globals.gelpiaID += 1
	# print("Constr?", Globals.enable_constr, " Begining New gelpia query->ID:", Globals.gelpiaID)

	# Combining conditional expression and external constraints
	str_constraint = " && ".join(
		[conditional_expression_string] + ([] if external_constraints_string is None or
												 len(external_constraints_string) == 0
										   else [external_constraints_string]))

	# Forming final input string for gelpia
	if Globals.enable_constr:
		gelpia_input_string = input_string + str_constraint + "; " + string_expression
	else:
		gelpia_input_string = input_string + string_expression  ## without the constraints

	# Logging gelpia queries
	if Globals.argList.gverbose:
		fout = open("gelpia_" + str(Globals.gelpiaID) + ".txt", "w")
		fout.write("# --input-epsilon {ieps}\n".format(ieps=str(gelpia_input_epsilon)))
		fout.write("# --output-epsilon {oeps}\n".format(oeps=str(gelpia_output_epsilon)))
		fout.write("# --output-epsilon-relative {oreps}\n".format(oreps=str(gelpia_output_epsilon_relative)))
		fout.write("# --dreal-epsilon {oeps}\n".format(oeps=str(gelpia_dreal_epsilon)))
		fout.write("# --dreal-epsilon-relative {oreps}\n".format(oreps=str(gelpia_dreal_epsilon_relative)))
		fout.write("# --timeout {tout}\n".format(tout=str(gelpia_timeout)))
		fout.write("# --max-iters {miters}\n".format(miters=str(gelpia_max_iters)))
		fout.write("{x3opt}".format(x3opt="# --use-z3\n" if Globals.argList.useZ3 else ""))
		fout.write(gelpia_input_string)
		fout.close()

	start_time = time.time()

	if max:
		extremum_function = gelpia.find_max
	else:
		extremum_function = gelpia.find_min
	extrema_lower_bound, extrema_upper_bound, inputs, solver_calls = extremum_function(gelpia_input_string,
															 gelpia_epsilons,
															 gelpia_timeout,
															 gelpia_grace,
															 gelpia_update,
															 gelpia_max_iters,
															 gelpia_seed,
															 False,
															 Globals.argList.useZ3,
															 gelpia.SRC_DIR,
															 gelpia_rust_executable,
															 False)
	# print("Inputs")
	# print(inputs)

	end_time = time.time()

	# Gelpia output stats
	# print("Finishing gelpia query->ID:", Globals.gelpiaID)
	# print("extrema_lower_bound", extrema_lower_bound, type(extrema_lower_bound))
	# print("extrema_upper_bound", extrema_upper_bound, type(extrema_upper_bound))
	# print("solver_calls", total_solver_calls)

	Globals.solver_calls += solver_calls

	if not max and extrema_lower_bound != "Overconstrained":
		return extrema_lower_bound
	elif max and extrema_upper_bound != "Overconstrained":
		return extrema_upper_bound
	else:
		0.0
	
#def extract_input_dep(free_syms):
#	ret_list = list()
#	for fsyms in free_syms:
#		ret_list += [str(fsyms), " = ", str(Globals.inputVars[fsyms]["INTV"]), ";"]
#	return "".join(ret_list)
#    #for name,val in inputs.items():
#    #    ret_list += [name, " = ", str(val["INTV"]), ";"]
#    #return "".join(ret_list)
def extract_input_dep(free_syms):
	ret_list = list()
	flist = [str(i) for i in free_syms]
	flist.sort()
	for fsyms in flist:
		ret_list += [str(fsyms), " = ", str(Globals.inputVars[seng.var(fsyms)]["INTV"]), ";"]
	return "".join(ret_list)
    #for name,val in inputs.items():
def genSig(sym_expr):
	try:
		if seng.count_ops(sym_expr) == 0 :
			print("Gensig:", sym_expr)
			return float(str(sym_expr))
	except ValueError:
		pass
	d = OrderedDict()
	flist = [str(i) for i in sym_expr.free_symbols]
	flist.sort()
	freeSyms = [seng.var(fs) for fs in flist]
	# make this to a map
	#for i in range(0, len(freeSyms)):
	#	inp = freeSyms[i]
	#	d[inp] = str(i)+"_"+"{intv}".format(intv=Globals.inputVars[inp]["INTV"])

	fpt = map(lambda i : (str(freeSyms[i]), str(i)+"_"+"{intv}".format(intv=Globals.inputVars[freeSyms[i]]["INTV"])), \
	                      range(len(freeSyms)))
	d =	{p[0]:p[1] for p in fpt}

	regex = re.compile("(%s)" % "|".join(map(re.escape, d.keys())))

	try:
		strSig = regex.sub(lambda mo: d[mo.string[mo.start():mo.end()]], str(sym_expr))
	except:
		print("Here:", sym_expr)
		sys.exit()

	return hashSig(strSig, "md5")

#--def generate_signature(sym_expr, cond_expr, externConstraints, cond_free_symbols, inputStr=None):
#--	print("GenSig:", sym_expr)
#--	try:
#--		if(seng.count_ops(sym_expr)==0):
#--			const_intv = float(str(sym_expr))
#--			return [const_intv, const_intv]
#--	except ValueError:
#--	    pass
#--
#--	hbs = len(Globals.hashBank.keys())
#--	#s2 = time.time()
#--	#print("\nTime for hashing sig = ", s2 - s1)
#--	#print("************ HBS : ", hbs, " ******************")
#--	if(hbs > 100):
#--		list(map(lambda x : Globals.hashBank.popitem(x) , list(Globals.hashBank.keys())[0:int(hbs/2)]))
#--	sig = genSig(sym_expr)
#--	check = Globals.hashBank.get(sig, None)
#--	if check is None:
#--		inputStr = inputStr if inputStr is not None else \
#--		            extract_input_dep(list(sym_expr.free_symbols.union(cond_free_symbols)))
#--		print("Gelpia input expr ops ->", seng.count_ops(sym_expr))
#--		print("InputStr({gelpiaid}):".format(gelpiaid=Globals.gelpiaID+1), inputStr)
#--		print("expression symbols :", sym_expr.free_symbols)
#--		print("cond symbols :", cond_free_symbols)
#--		g1 = time.time()
#--		Globals.hashBank[sig] = invoke_gelpia(sym_expr, cond_expr, externConstraints, inputStr)
#--		g2 = time.time()
#--		print("Gelpia solve = ", g2 - g1, "opCount =", seng.count_ops(sym_expr))
#--	else:
#--		#print("MATCH FOUND")
#--		#Globals.hashBank[sig] = check
#--		print("Just passing")
#--		pass
#--
#--	return Globals.hashBank[sig]


def generate_signature(sym_expr, cond_expr, externConstraints, cond_free_symbols, inputStr=None):
	print("GenSig:", sym_expr)
	try:
		if(seng.count_ops(sym_expr)==0):
			const_intv = float(str(sym_expr))
			return [const_intv, const_intv]
	except ValueError:
	    pass
	inputStr = inputStr if inputStr is not None else \
	            extract_input_dep(list(sym_expr.free_symbols.union(cond_free_symbols)))
	#print("Gelpia input expr ops ->", seng.count_ops(sym_expr))
	#print("InputStr({gelpiaid}):".format(gelpiaid=Globals.gelpiaID+1), inputStr)
	#print("expression symbols :", sym_expr.free_symbols)
	#print("cond symbols :", cond_free_symbols)
	g1 = time.time()
	val = invoke_gelpia(sym_expr, cond_expr, externConstraints, inputStr)
	g2 = time.time()
	print("\tGelpia solve = {duration}, opCount = {ops}".format(duration=g2 - g1, ops=seng.count_ops(sym_expr)))
	return val

def statistical_eval( symDict, expr , niters=10000 ):

	res = 0.0
	maxres = 0.0
	for t in range(niters):
		vecDict = {k:random.uniform(v[0],v[1]) for k, v in symDict.items()}
		val = expr.subs(vecDict) 
		res = res + val  
		maxres = max(val,maxres) 

	#return [mean-sample-error, max-sample-error]
	#print("From statistical eval:", (res/niters, maxres))
	return (res/niters, maxres)

## Performs statistical sampling of the symbolic expression
## Used for obtaining statistical profile on error distribution
def get_statistics(sym_expr, inputDict=None):

	try:
		if(seng.count_ops(sym_expr)==0):
			const_val = float(str(sym_expr))
			return (const_val, const_val)
	except ValueError:
		pass

	symDict = inputDict if inputDict is not None else {fsyms : Globals.inputVars[fsyms]["INTV"] for fsyms in sym_expr.free_symbols}
	res_avg_maxres = statistical_eval ( symDict, sym_expr, niters=Globals.argList.samples)
	return res_avg_maxres


def isConst(obj):
	"""
	Checks if node represents a constant. We only have Num nodes representing constants

	Parameters
	----------
	obj : node type
		Any node.

	Returns
	-------
	bool
		Returns True if node is Num else False
	"""
	if type(obj).__name__ == "Num":
		return True
	else:
		return False
		#try:
		#	x = float(str(obj.f_expression))
		#	return True
		#except:
		#	return False

# Partitions "items" into two lists. One that satisfied "predicate" and the other that does not.
def partition(items, predicate):
	a, b = tee((predicate(item), item) for item in items)
	return ((item for pred, item in a if not pred), (item for pred, item in b if pred))


def extract_partialAST(NodeList):

	parent_dict = dict()

	max_depth = max(list(map(lambda x: x.depth, NodeList)))
	it1, it2 = partition(NodeList, lambda x:x.depth==max_depth)
	next_workList = list(it1)
	workList = list(it2)

	for w in NodeList:
		parent_dict[w] = []

	# the assumption is all previous depths are seen by now
	while(len(workList) > 0): 
		node = workList.pop(0)
		for child in node.children:
			parent_dict[child] = parent_dict.get(child, [])
			parent_dict[child].append(node)
			next_workList.append(child)

		curr_depth = node.depth
		next_depth = curr_depth - 1
		if(len(workList)==0 and next_depth != -1 and len(next_workList)!=0):
			nextIter, currIter = partition(next_workList, \
				                                     lambda x:x.depth==next_depth)
			workList = list(set(currIter))
			next_workList = list(set(nextIter))

	## debug check
	#for k, vlist in parent_dict.items():
	#	print(k.f_expression, [v.f_expression for v in vlist])

	return parent_dict


def binary_search_on_input_var(symbolic_expression, conditional_expression, external_constraints, input_interval_dict,
							   extremum, input_variable, max=True):
	"""
	[Recursive] Binary search method to find the narrowest interval for input_variable corresponding to extremum of
	symbolic_expression.

	Algorithm
	---------
	Return original interval if input variable interval is smaller than some threshold
	Select lower half of interval

	if the extremum changes on lower half
		recurse on upper half and return result
	if extremum remains the same on upper half
		return original interval

	Parameters
	----------
	symbolic_expression : TODO: Fill this
		Symbolic expression that gives the optimal value.
	conditional_expression : TODO: Fill this
		Conditional expression that acts as a constraint on the domain
	external_constraints : TODO: Fill this
		Constraints provided by the user that may narrow the domain
	input_interval_dict : dict
		Dictionary(input_variable -> interval_tuple)
	extremum : float
		Optimal value of symbolic_expression within given domain following all constraints.
	input_variable : TODO: Fill this
		Input variable being worked on
	max : bool
		Determines whether to find max or min value of symbolic_expression

	Returns
	-------
	input_interval_dict : dict
		Dictionary(input_variable -> interval_tuple) of inputs with the newly found narrowest interval for input_variable
	"""
	# Threshold to stop reducing interval size and quit search. Ideally should be machine epsilon for the float type used.
	interval_difference_threshold = 2**-51

	# Base case of recursion: If interval is smaller than specified threshold
	if input_interval_dict[input_variable][1] - input_interval_dict[input_variable][0] < interval_difference_threshold:
		return input_interval_dict

	# Mid-value
	interval_mid = (input_interval_dict[input_variable][0]+input_interval_dict[input_variable][1]) / 2
	# print(input_interval_dict)

	# For loop for the two halves of interval, starting with the lower half, so we modify the upper bound (index 1) first.
	for index in range(2)[::-1]:
		# Deep copying input_interval_dict since we want to retain the original candidate_interval_dict
		candidate_interval_dict = copy.deepcopy(input_interval_dict)

		# Selecting Upper/Lower half of interval by modifying the interval bound appropriately. index=1 selects lower half
		# in which case we subtract the smallest interval difference as well to avoid common values in upper and lower halves
		candidate_interval_dict[input_variable][index] = interval_mid
		if index == 1:
			candidate_interval_dict[input_variable][index] -= interval_difference_threshold

		input_variable_string_list = []
		for variable, interval in candidate_interval_dict.items():
			input_variable_string_list += [variable, " = ", str(interval), ";"]
		input_string = "".join(input_variable_string_list)

		# Computing extremum of narrowed interval
		print("Input sent")
		print(candidate_interval_dict)
		computed_extremum = extremum_of_symbolic_expression(symbolic_expression, "<<True>>", "", input_string, True)
		print(computed_extremum)
		print(interval_mid)

		# Check to ensure the extremum does not beat the original extremum
		assert computed_extremum <= extremum if max else computed_extremum >= extremum
		# Check to ensure the extremum remains the same for the upper half
		if not index:
			assert computed_extremum == extremum if max else computed_extremum == extremum

		# Recursive section
		# If lower half changes extremum, we recurse on upper half else we continue to the next iteration (Checking if
		# upper half changes extremum).
		# NOTE: It can NEVER happen that upper half changes extremum. Since upper half is checked only when lower half
		# changes the extremum. So the block of code inside IF condition should NEVER execute for the 2nd iteration.
		# The above assert statement guards this case.
		if (max and computed_extremum < extremum) or (not max and computed_extremum > extremum):
			candidate_interval_dict = copy.deepcopy(input_interval_dict)
			# Modifying lower bound of interval (selects upper half)
			candidate_interval_dict[input_variable][int(not index)] = interval_mid
			return binary_search_on_input_var(symbolic_expression, conditional_expression, external_constraints,
											  candidate_interval_dict, extremum, input_variable, max=True)

	return input_interval_dict


def binary_search_for_input_set(symbolic_expression, conditional_expression, external_constraints, input_interval_dict,
								extremum, max=True):
	"""
	Searches for narrowest box of input intervals giving optimal value of 'symbolic_expression' using binary search
	method on each input variable.

	Parameters
	----------
	symbolic_expression : TODO: Fill this
		Symbolic expression that gives the optimal value.
	conditional_expression : TODO: Fill this
		Conditional expression that acts as a constraint on the domain
	external_constraints : TODO: Fill this
		Constraints provided by the user that may narrow the domain
	input_interval_dict : dict
		Dictionary(input_variable -> interval_tuple)
	extremum : float
		optimal value of symbolic_expression within given domain following all constraints.
	max : bool
		Determines whether to find max or min value of symbolic_expression

	Returns
	-------
	(Symbol, tuple)
		Dictionary(input_variable -> interval_tuple) of narrowest box of inputs
	"""
	candidate_interval_dict = input_interval_dict

	# Performing binary search for optimal value on each input variable, propagating the narrowest intervals found to
	# the next iteration
	for input_variable, interval in input_interval_dict.items():
		print(input_variable)
		# print(candidate_interval_dict)
		candidate_interval_dict = binary_search_on_input_var(symbolic_expression, conditional_expression, external_constraints, candidate_interval_dict, extremum, input_variable, max)

	return candidate_interval_dict
