#!/usr/bin/python3
import  sys
import os
import time
import argparse
import symengine as seng

import Globals
from gtokens import *
from lexer import Slex
from parser import Sparser

from collections import defaultdict

from ASTtypes import *

import helper
from AnalyzeNode_Cond import AnalyzeNode_Cond

import logging

def parseArguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', help='Test file name', required=True)
	parser.add_argument('--parallel', help='Enable parallel optimizer queries:use for large ASTs',\
							default=False, action='store_true')
	parser.add_argument('--enable-abstraction', help='Enable abstraction of internal node defintions,\
													  value indiactes level\
													  of abstraction. By default enabled to level-1. \
													  To disable pass 0', default=False, action='store_true')
	parser.add_argument('--mindepth', help='Min depth for abstraction. Default is 10',\
									  default=10, type=int)
	parser.add_argument('--maxdepth', help='Max depth for abstraction. Limiting to 40', \
									  default=40, type=int)
	parser.add_argument('--stat', help='Report statistics on error expression. Disabled by default', \
									  default=False, action='store_true')
	parser.add_argument('--samples', help='Number of samples for stats', \
									  default=1000, type=int)
	parser.add_argument('--report-instability', help='Report instability for every divergent path. Disabled by default', \
									  default=False, action='store_true')
	parser.add_argument('--simplify', help='Simplify expression -> could be costly for very large expressions',
										default=False, action='store_true')
	parser.add_argument('--logfile', help='Python logging file name -> default is default.log', default='default.log')
	parser.add_argument('--outfile', help='Name of the output file to write error info', default='outfile.txt')
	parser.add_argument('--std', help='Print the result to stdout', default=False, action='store_true')
	parser.add_argument('--sound', help='Turn on analysis for higher order errors', default=False, action='store_true')
	parser.add_argument('--compress', help='Perform signature matching to reduce optimizer calls using hashing and md5 signature', default=False, action='store_true')
	parser.add_argument('--force', help='Sideline additional tricks used for non-linear examples. Use this option for linear examples', default=False, action='store_true')
	parser.add_argument('--realpaver', help='Enable realpaver guided subdivision inside satire+', default=False, action='store_true')
	parser.add_argument('--enable-constr', help='Enable solving constrained optimization queries', default=False, action='store_true')
	parser.add_argument('--stat-err-enable', help='Enable statistical error sampling', default=False, action='store_true')
	parser.add_argument('--stat-err-fraction', help='Fractional bound for using statistical error. Default is 0.5',\
	                                  default=0.5, type=float)
	parser.add_argument('--useZ3', help='Enabled using Z3 for constraint solving. Dreal gets disabled', default=False, action='store_true')
	parser.add_argument('--stable', help='Inform to ignore instability correction', default=False, action='store_true')
	parser.add_argument('--gverbose', help='Create dumps of each optimizer query for debugging', default=False, action='store_true')
	                                  

	result = parser.parse_args()
	return result


def rebuildASTNode(node, completed):
	
	for child in node.children:
		if not completed.__contains__(child):
			rebuildASTNode(child, completed)

	node.depth = 0 if len(node.children)==0 or type(node).__name__ == "FreeVar" \
				   else max([child.depth for child in node.children]) +1

	if node.token.type in ops.DFOPS_LIST:
		completed[node] = node.depth


def rebuildAST():
	rb1 = time.time()
	print("\n  ********* Rebuilding AST post abstracttion ********\n")
	logger.info("\n  ********* Rebuilding AST post abstracttion ********\n")
	print("Synthesizing expression with fresh FreeVars .... ")
	probeList = mod_probe_list([Globals.global_symbol_table[0]._symTab[outVar] for outVar in Globals.outVars])

	completed = defaultdict(int)

	Globals.simplify=True  ## Enable simplify to begin
	for csym, symNode in Globals.predTable.items():
		print("OUT:Ever came here1?")
		rebuildASTNode(symNode, completed)
		print("OUT:CSYM:", csym, symNode.depth)

	for node in probeList:
		if not completed.__contains__(node):
			rebuildASTNode(node, completed)

	maxdepth = max([node.depth for node in probeList])
	#print("**New maxdepth = ", maxdepth)
	beforetotalNodes = sum([len(v) for k,v in Globals.depthTable.items()])
	Globals.depthTable = {depth : set([node for node in completed.keys() if node.depth==depth]) for depth in range(maxdepth+1)}

	aftertotalNodes = sum([len(v) for k,v in Globals.depthTable.items()])

	print(" >  Rebuilt AST Nodes = {new_num_nodes} ({old_num_nodes})".format(old_num_nodes=beforetotalNodes, new_num_nodes=aftertotalNodes))
	print(" >  Rebuilt AST Depth = {ast_depth}".format(ast_depth=maxdepth))
	logger.info(" >  Rebuilt AST Nodes = {new_num_nodes} ({old_num_nodes})".format(old_num_nodes=beforetotalNodes, new_num_nodes=aftertotalNodes))
	logger.info(" >  Rebuilt AST Depth = {ast_depth}".format(ast_depth=maxdepth))
	rb2 = time.time()
	print("  **********Rebuilt AST in {duration} secs********\n".format(duration=rb2-rb1))
	logger.info("  **********Rebuilt AST in {duration} secs********\n".format(duration=rb2-rb1))


def abstractNodes(results):

	
	
	for node, res in results.items():
		Globals.FID += 1
		name = seng.var("FR"+str(Globals.FID))
		node.__class__ = FreeVar
		node.children = ()
		node.depth = 0

		node.set_noise(node, (res["ERR"], res["SERR"]))
		node.mutate_to_abstract(name, ID)

		Globals.inputVars[name] = {"INTV" : res["INTV"]}
		Globals.global_symbol_table[0]._symTab[name] = ((node,Globals.__T__),)
	

def simplify_with_abstraction(sel_candidate_list, argList, maxdepth, final=False):

	Globals.condExprBank.clear()
	obj = AnalyzeNode_Cond(sel_candidate_list, argList, maxdepth, paving=argList.realpaver)
	print("Start!")
	results = obj.start()

	if "flag" in results.keys():
		print("Returned w/o execution-->need to modify bound")
		return results

	del obj
	if final:
		#for k,v in results.items():
		#	print(v["ERR"]*pow(2,-53), v["INTV"])
		return results

	abstractNodes(results)
	rebuildAST()

	


	return dict()





def full_analysis(probeList, argList, maxdepth):
	#helper.expression_builder(probeList)
	#for k,v in Globals.predTable.items():
	#	print(k,v)
	#obj = AnalyzeNode_Cond(probeList, argList, maxdepth)
	#obj.start()
	print("\n-----------------------------------")
	print("Full Analysis Block:\n")
	res = simplify_with_abstraction(probeList, argList, maxdepth,final=True)
	print("-----------------------------------\n")
	print(res)
	return res
	

#def ErrorAnalysis(argList):
#
#	probeList = [Globals.global_symbol_table[0]._symTab[outVar] for outVar in Globals.outVars]()
#	maxdepth = max([max([n[0].depth for n in nodeList])  for nodeList in probeList])
#	print("maxdepth = ", maxdepth)
#	probeList = [nodeList[0][0] for nodeList in probeList]
#
#	## Check on the conditonal nodes----------
#	## ---------------------------------------
#	#for k,v in Globals.predTable.items():
#	#	print(k, v.rec_eval(v), type(v).__name__)
#
#	#for k,v in Globals.condTable.items():
#	#	print(k, v.rec_eval(v))
#
#
#
#	full_analysis(probeList, argList, maxdepth)

def mod_probe_list(probeNodeList):
	probeList = [Globals.global_symbol_table[0]._symTab[outVar] for outVar in Globals.outVars]
	probeList = [nodeList[0][0] for nodeList in probeList]
	return probeList
	

	#opList = helper.get_opList(DIV)
	#D = helper.find_common_dependence(opList, 5, 40)
	#for k,v in D.items():
	#	print(v)
	#	for dep in v:
	#		print(k.depth, dep.depth, dep.rec_eval(dep))
	#print("From here:", [op.rec_eval(op) for op in opList])
def ErrorAnalysis(argList):
	abstraction_level = 1
	probeList = [Globals.global_symbol_table[0]._symTab[outVar] for outVar in Globals.outVars]
	maxdepth = max([max([n[0].depth for n in nodeList])  for nodeList in probeList])
	print("maxdepth = ", maxdepth)
	logger.info("Full AST_DEPTH : {ast_depth}".format(ast_depth=maxdepth))
	probeList = [nodeList[0][0] for nodeList in probeList]
	bound_mindepth, bound_maxdepth = argList.mindepth, argList.maxdepth


	if ( argList.enable_abstraction ) :
		print("\nAbstraction Enabled... \n")
		logger.info("\nAbstraction Enabled... \n")
		print(bound_maxdepth, bound_mindepth, maxdepth)
		while ( maxdepth >= bound_maxdepth and maxdepth >= bound_mindepth ):
			[abs_depth, sel_candidate_list] = helper.selectCandidateNodes(maxdepth, bound_mindepth, bound_maxdepth)
			print("Candidate List Length:", len(sel_candidate_list))
			if ( len(sel_candidate_list) > 0):
				print("-----------------------------------")
				print("ABSTRACTION LEVEL = {abs_level}".format(abs_level=abstraction_level))
				logger.info("-----------------------------------")
				logger.info("ABSTRACTION LEVEL = {abs_level}".format(abs_level=abstraction_level))
				abstraction_level += 1
				results = simplify_with_abstraction(sel_candidate_list, argList, maxdepth)
				maxopCount = results.get("maxOpCount", 1000)
				probeList = mod_probe_list([Globals.global_symbol_table[0]._symTab[outVar] for outVar in Globals.outVars])
				maxdepth = max([n.depth for n in probeList])
				if (maxopCount > 1000 and maxdepth > 8 and bound_mindepth > 5):
					bound_maxdepth = maxdepth if bound_maxdepth > maxdepth else bound_maxdepth - 2 if bound_maxdepth - bound_mindepth > 4 else bound_maxdepth
					bound_mindepth = bound_mindepth - 2 if bound_maxdepth - bound_mindepth > 4 else bound_mindepth
				elif maxdepth <= bound_maxdepth and maxdepth > bound_mindepth:
					bound_maxdepth = maxdepth
					assert(bound_maxdepth >= bound_mindepth)
				print("-----------------------------------\n")
				logger.info("-----------------------------------\n")
			else:
				break
		return full_analysis(probeList, argList, maxdepth)
	else:
		return full_analysis(probeList, argList, maxdepth)




if __name__ == "__main__":
	start_exec_time	= time.time()
	argList = parseArguments()
	Globals.argList = argList
	Globals.enable_constr = argList.enable_constr
	sys.setrecursionlimit(10**6)
	print("LEVEL_TOP: Parsed argList = ", argList)
	text = open(argList.file, 'r').read()
	fout = open(argList.outfile, 'w')


	#--------- Setup Logger ------------------------------------
	logging.basicConfig(filename=argList.logfile,
					level = logging.INFO,
					filemode = 'w')
	logger = logging.getLogger()

	##-------- Check if realpaver is available -----------------
	Globals.ROOT_DIR = os.getenv("SAT_ROOT")
	assert ( os.path.isdir(Globals.ROOT_DIR) )
	if argList.realpaver:
		Globals.LIBFILE = Globals.ROOT_DIR+"/RL1/build/"+"libsatrp.so"
		assert ( os.path.isfile(Globals.LIBFILE) )



	#--------- Lexing and Parsing ------------------------------
	start_parse_time = time.time()
	lexer = Slex()
	parser = Sparser(lexer)
	parser.parse(text)
	del parser
	del lexer
	end_parse_time = time.time()
	parse_time = end_parse_time - start_parse_time
	logger.info("Parsing time : {parse_time} secs".format(parse_time = parse_time))
	print("LEVEL_TOP: End of Parsing: Parsing time : {parse_time} secs".format(parse_time = parse_time))



	#print("Before:", Globals.global_symbol_table[0]._symTab.keys())
	#------ PreProcess to eliminate all redundant nodes ---------
	pr1 = time.time()
	#helper.PreProcessAST()
	pr2 = time.time()
	##print("\nAfter:", Globals.global_symbol_table[0]._symTab.keys(),"\n\n")
	#opList = helper.get_opList(DIV)
	#D = helper.find_common_dependence(opList, 5, 40)
	#for k,v in D.items():
	#	print(v)
	#	for dep in v:
	#		print(k.depth, dep.depth, dep.rec_eval(dep))
	#print("From here:", [op.rec_eval(op) for op in opList])



	#------ Main Analysis ----------------------------------------
	ea1 = time.time()
	results = ErrorAnalysis(argList)
	ea2 = time.time()

	
	#------ Write to File -----------------------------------------
	helper.writeToFile(results, fout, argList)
	fout.close()

	end_exec_time = time.time()
	full_time = end_exec_time - start_exec_time 

	logger.info("Optimizer calls : {num_calls}\n".format(num_calls = Globals.gelpiaID))
	logger.info("Smt calls : {num_calls}\n".format(num_calls = Globals.solver_calls))
	logger.info("Parsing time : {parsing_time}\n".format(parsing_time = parse_time))
	logger.info("PreProcessing time : {preprocess_time}\n".format(preprocess_time = pr2-pr1))
	logger.info("Analysis time : {analysis_time}\n".format(analysis_time = ea2-ea1))
	logger.info("Full time : {full_time}\n".format(full_time = full_time))

	print("LEVEL_TOP: Optimizer calls : {num_calls}".format(num_calls = Globals.gelpiaID))
	print("LEVEL_TOP: Smt calls : {num_calls}".format(num_calls = Globals.solver_calls))
	print("LEVEL_TOP: Parsing time : {parsing_time}".format(parsing_time = parse_time))
	print("LEVEL_TOP: PreProcessing time : {preprocess_time}".format(preprocess_time = pr2-pr1))
	print("LEVEL_TOP: Analysis time : {analysis_time}".format(analysis_time = ea2-ea1))
	print("LEVEL_TOP: Full time : {full_time}".format(full_time = full_time))
	
	print(Globals.argList.stat_err_enable, Globals.argList.stat_err_fraction)

	## --- some extra profiling data
	#for k,v in Globals.InstabID.items():
	#	print(type(k), k.f_expression,v)
	#print("Hello:", Globals.InstabID)

	#print("\n\n")

	#for k, v in Globals.InstabDict.items():
	#	print(k, v)

	D = list(set([(( k[0].exprCond[1], k[1].exprCond[1]),v[0]) for k,v in Globals.InstabDict.items() if v is not None and v[0] > 0.0]))
	D.sort(key=lambda tup: -tup[1])
	print("Printing sorted instability list:")
	for i in range(len(D)):
		print("Pred:{pred}\t instab:{instab}".format(pred=D[i][0], instab=D[i][1]))


	D1 = dict()
	for k,v in Globals.InstabDict.items():
		if v is not None and v[0] > 0.0:
			KEY = ( k[0].exprCond[1], k[1].exprCond[1])
			entry = D1.get(KEY, [])
			entry.append([v[0], (k[0].exprCond[0], k[1].exprCond[0], k)])
			D1[KEY] = entry
			
	for k,v in D1.items():
		D1[k].sort(key=lambda tup: -tup[0])

	# D = [(instab, (expr1, expr2), pred),...]
	D = [ (v[0][0], v[0][1], k) for k,v in D1.items() ]
	D.sort(key=lambda tup: -tup[0])
	print("Printing sorted instability list:")
	for i in range(len(D)):
		print("Pred:{pred}\t instab:{instab}".format(pred=D[i][2], instab=D[i][0]))


