
import  sys
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
	#parser.add_argument('--fixdepth', help='Fix the abstraction depth. Default is -1(disabled)', \
	#								  default=-1, type=int)
	#parser.add_argument('--alg', help='Heuristic level for abstraction(default 0), \
	#									0 -> Optimal depth within mindepth and maxdepth(may loose correlation), \
	#									1 -> User defined fixed depth, \
	#									2 -> User Tagged nodes or only lhs nodes', \
	#									default=0, type=int)
	parser.add_argument('--simplify', help='Simplify expression -> could be costly for very large expressions',
										default=False, action='store_true')
	parser.add_argument('--logfile', help='Python logging file name -> default is default.log', default='default.log')
	parser.add_argument('--outfile', help='Name of the output file to write error info', default='outfile.txt')
	parser.add_argument('--std', help='Print the result to stdout', default=False, action='store_true')
	parser.add_argument('--sound', help='Turn on analysis for higher order errors', default=False, action='store_true')
	parser.add_argument('--compress', help='Perform signature matching to reduce optimizer calls using hashing and md5 signature', default=False, action='store_true')
	parser.add_argument('--force', help='Sideline additional tricks used for non-linear examples. Use this option for linear examples', default=False, action='store_true')
	                                  

	result = parser.parse_args()
	return result


def rebuildASTNode(node, completed):
	
	for child in node.children:
		if not completed.__contains__(child):
			rebuildASTNode(child, completed)

	node.depth = 0 if len(node.children)==0 or type(node).__name__ == "FreeVar" \
				   else max([child.depth for child in node.children]) +1
	completed[node] = node.depth


def rebuildAST():
	rb1 = time.time()
	print("\n  ********* Rebuilding AST post abstracttion ********\n")
	logger.info("\n  ********* Rebuilding AST post abstracttion ********\n")
	print("Synthesizing expression with fresh FreeVars .... ")
	probeList = mod_probe_list(helper.getProbeList())

	completed = defaultdict(int)

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
		name = seng.var("_F"+str(Globals.FID))
		node.__class__ = FreeVar
		node.children = ()
		node.depth = 0

		node.set_noise(node, (res["ERR"], res["SERR"]))
		node.mutate_to_abstract(name, ID)

		Globals.inputVars[name] = {"INTV" : res["INTV"]}
		Globals.GS[0]._symTab[name] = ((node,Globals.__T__),)
	

def simplify_with_abstraction(sel_candidate_list, argList, maxdepth, final=False):


	obj = AnalyzeNode_Cond(sel_candidate_list, argList, maxdepth)
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
	return res
	

#def ErrorAnalysis(argList):
#
#	probeList = helper.getProbeList()
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
	probeList = helper.getProbeList()
	probeList = [nodeList[0][0] for nodeList in probeList]
	return probeList
	

def ErrorAnalysis(argList):

	absCount = 1
	probeList = helper.getProbeList()
	maxdepth = max([max([n[0].depth for n in nodeList])  for nodeList in probeList])
	print("maxdepth = ", maxdepth)
	logger.info("Full AST_DEPTH : {ast_depth}".format(ast_depth=maxdepth))
	probeList = [nodeList[0][0] for nodeList in probeList]
	bound_mindepth, bound_maxdepth = argList.mindepth, argList.maxdepth


	if ( argList.enable_abstraction ) :
		print("\nAbstraction Enabled... \n")
		logger.info("\nAbstraction Enabled... \n")
		while ( maxdepth >= bound_maxdepth and maxdepth >= bound_mindepth ):
			[abs_depth, sel_candidate_list] = helper.selectCandidateNodes(maxdepth, bound_mindepth, bound_maxdepth)
			print("Candidate List Length:", len(sel_candidate_list))
			if ( len(sel_candidate_list) > 0):
				print("-----------------------------------")
				print("ABSTRACTION LEVEL = {abs_level}".format(abs_level=absCount))
				logger.info("-----------------------------------")
				logger.info("ABSTRACTION LEVEL = {abs_level}".format(abs_level=absCount))
				absCount += 1
				results = simplify_with_abstraction(sel_candidate_list, argList, maxdepth)
				maxopCount = results.get("maxOpCount", 1000)
				probeList = mod_probe_list(helper.getProbeList())
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
	sys.setrecursionlimit(10**6)
	print(argList)
	text = open(argList.file, 'r').read()
	fout = open(argList.outfile, 'w')


	#--------- Setup Logger ------------------------------------
	logging.basicConfig(filename=argList.logfile,
					level = logging.INFO,
					filemode = 'w')
	logger = logging.getLogger()



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



	#print("Before:", Globals.GS[0]._symTab.keys())
	#------ PreProcess to eliminate all redundant nodes ---------
	pr1 = time.time()
	helper.PreProcessAST()
	pr2 = time.time()
	#print("\nAfter:", Globals.GS[0]._symTab.keys(),"\n\n")



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
	logger.info("Parsing time : {parsing_time}\n".format(parsing_time = parse_time))
	logger.info("PreProcessing time : {preprocess_time}\n".format(preprocess_time = pr2-pr1))
	logger.info("Analysis time : {analysis_time}\n".format(analysis_time = ea2-ea1))
	logger.info("Full time : {full_time}\n".format(full_time = full_time))

	print("Optimizer calls : {num_calls}".format(num_calls = Globals.gelpiaID))
	print("Parsing time : {parsing_time}".format(parsing_time = parse_time))
	print("PreProcessing time : {preprocess_time}".format(preprocess_time = pr2-pr1))
	print("Analysis time : {analysis_time}".format(analysis_time = ea2-ea1))
	print("Full time : {full_time}".format(full_time = full_time))
	
