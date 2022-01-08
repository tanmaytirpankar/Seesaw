
import Globals
import math
import sympy as sym
import symengine as seng
from collections import defaultdict
from SymbolTable import *

from gtokens import *

import ops_def as ops

import logging 

import ctypes

logger = logging.getLogger(__name__)
from AnalyzeNode_Cond import AnalyzeNode_Cond as ANC


class Cintervals(ctypes.Structure):
	_fields_ = ("name", ctypes.c_char_p), ("x", ctypes.c_double), ("y", ctypes.c_double)


def rpInterface(rpConstraint, numVars, numBoxes):

	print("IN-RPCONSTR:", rpConstraint)
	#rp = ctypes.CDLL("RL1/build/libsatrp.so")
	rp = ctypes.CDLL(Globals.LIBFILE)
	rp.initializeRP.restype = ctypes.POINTER(ctypes.POINTER(Cintervals * numVars)*numBoxes)
	print("@D BOX size -> ", numVars * numBoxes)
	rp.initializeRP.argtypes = [ctypes.c_char_p]

	returnValue = rp.initializeRP(rpConstraint.encode())

	#print(rpConstraint)

	#print("From RP:", returnValue)
	#for box in returnValue.contents:
	#	for var in box.contents:
	#		print("==========")
	#		print(var.name.decode())
	#		print(var.x)
	#		print(var.y)

	return [returnValue, rp._handle]


def getProbeList():
	#print(Globals.GS[0]._symTab.keys())
	return [Globals.GS[0]._symTab[outVar] for outVar in Globals.outVars]

def freeCondSymbols(SymEl):
	assert(type(SymEl).__name__ == "Sym")
	cond = SymEl.exprCond[1]
	if cond in (True,False):
		return set()
	else:
		return cond.free_symbols


def parse_cond(cond):
	tcond = cond
	if tcond not in (True,False):
		free_syms = tcond.free_symbols
		for fsym in free_syms:
			symNode = Globals.predTable[fsym]
			#print(fsym," |-> ", symNode.rec_eval(symNode))
			subcond =  Globals.condExprBank.get(fsym, symNode.rec_eval(symNode))
			Globals.condExprBank[fsym] = subcond
		#tcond = tcond.subs({fsym:subcond})
		tcond = tcond.subs({fsym: Globals.condExprBank[fsym] for fsym in free_syms})
		#print("Cond, :-> ", tcond)
		return tcond
	return tcond


def dfs_expression_builder(node, reachable, parent_dict, free_syms, cond_syms, cond, etype, ctype, inv=False):

	for child in node.children:
		if not reachable[child.depth].__contains__(child):
			(free_syms, cond_syms) = dfs_expression_builder(child, reachable, parent_dict, free_syms, cond_syms, cond, etype, ctype, inv)

		parent_dict[child].append(node)

	if type(node).__name__ == "ExprComp":
		print("ExprComp line:", node.token.lineno)
		if etype:
			print("HIDDEN CONDITIONAL DEPTH:", node.children[0].depth, node.children[1].depth)
			res0 = ANC([node.children[0]], [], node.children[0].depth, Globals.argList.realpaver).start()
			res1 = ANC([node.children[1]], [], node.children[1].depth, Globals.argList.realpaver).start()
		## an exprComp node as a modified evaluation ops to include extra error terms
			(fexpr,fsyms) = node.mod_eval(node, inv, res0[node.children[0]]["ERR"]*pow(2,-53), \
								   res1[node.children[1]]["ERR"]*pow(2,-53) )
			print("Never ever error")
		else:
			(fexpr,fsyms) = node.mod_eval(node, inv, 0.0, 0.0)
		free_syms = free_syms.union(fsyms)
	else:
		fexpr = node.eval(node, inv)

	node.set_expression(fexpr)
	#print("FEXPRESSION TYPE = ", type(node.f_expression).__name__)
	if type(node.f_expression).__name__ == "SymTup":
		csymSet = reduce(lambda x,y: x.union(y), \
						[el.exprCond[1].free_symbols for el in node.f_expression if el.exprCond[1] not in (True,False)], \
						set())
		cond_syms = cond_syms.union(csymSet)
	reachable[node.depth].add(node)

	return (free_syms, cond_syms)


	#print(node.depth, type(node).__name__, node.cond)
	#print("main:", type(node).__name__, fexpr)
	#print(node.token)
	#print([(type(child).__name__, child.f_expression) for child in node.children], "\n\n")
	#fexpr = node.eval(node)
	#node.set_expression(fexpr)
	#reachable[node.depth].add(node)



def expression_builder(probeList, etype=False, ctype=False, inv=False):

	parent_dict = defaultdict(list)
	reachable = defaultdict(set)
	free_syms = set()
	cond_syms = set()
	# print("Begin enter")

	for node in probeList:
		if not reachable[node.depth].__contains__(node):
			# print(node.depth)
			(free_syms, cond_syms) = dfs_expression_builder(node, reachable, parent_dict, free_syms, cond_syms, cond=Globals.__T__,etype=etype, ctype=ctype, inv=inv)

		#print(node.f_expression)
		#print("From expression builder: Root node stats -> opcount={opcount}, depth={depth}".format(opcount=0 if type(node).__name__ not in ('TransOp', 'BinOp') else node.f_expression.__countops__(), depth=node.depth))
		#print(type(node).__name__, node.token.type, node.depth, node.f_expression)
		#print([(type(child).__name__, child.token.type, child.depth, child.f_expression) for child in node.children])

	del reachable

	#print("Inside expression builder :", cond_syms)
	
	# print("Begin exit")
	if ctype:
		return (free_syms, cond_syms)
	else:
		#for k,v in Globals.GS[0]._symTab.items():
		#	print("\n*******Symbol Name:", k)
		#	for vi in v:
		#		print(vi[0].f_expression)
		return (parent_dict, cond_syms)

#etype = to analyze error within the conditional expressions
#ctype = free_syms + cond_syms ( specifically used when handling conditonals to retrieve conditional symbols )
# inv  = To generate delta inverse that includes the grey-zone
def handleConditionals(probeNodeList, etype=True, inv=False):
	print("Building conditional expressions...\n")
	logger.info("Building conditional expressions...\n")
	(fsyms, csyms) = expression_builder(probeNodeList, etype, ctype=True, inv=inv)
	cstr = " & ".join([str(probeNode.f_expression) for probeNode in probeNodeList])
	print("Debug-check:", (cstr,fsyms, csyms))
	#return (" & ".join([str(probeNode.f_expression) for probeNode in probeNodeList]),fsyms, csyms)
	return (cstr,fsyms, csyms)

def pretraverse(node, reachable):
	
	for child in node.children:
		#print("child", child, type(node).__name__)
		if reachable[child.depth].__contains__(child):
			pass
		else:
			pretraverse(child, reachable)

	reachable[node.depth].add(node)

	## debug-check for node is dov and line no
	if node.token.type == DIV:
		print("DIV :-->", node.token.lineno)


def PreProcessAST():
	
	print("\n------------------------------")
	print("PreProcessing Block:")

	probeList = getProbeList()
	reachable = defaultdict(set)


	for nodeList in probeList:
		assert(len(nodeList)==1)
		[node,cond] = nodeList[0]
		if not reachable[node.depth].__contains__(node):
			pretraverse(node, reachable)

	print("Symbol count Pre Processing :", len(Globals.GS[0]._symTab.keys()))
	Globals.GS[0]._symTab = {syms: tuple(set(n for n in nodeCondList \
										if reachable[n[0].depth].__contains__(n[0]))) \
										for syms,nodeCondList in Globals.GS[0]._symTab.items() }
	print("Symbol count Post Processing :", len(Globals.GS[0]._symTab.keys()))
	prev_numNodes = sum([ len(Globals.depthTable[el]) for el in Globals.depthTable.keys() if el!=0] )
	Globals.depthTable = reachable
	curr_numNodes = sum([ len(Globals.depthTable[el]) for el in Globals.depthTable.keys() if el!=0] )
	logger.info("Total number of nodes pre-processing: {prev}".format(prev=prev_numNodes))
	logger.info("Total number of nodes post-processing: {curr}".format(curr=curr_numNodes))
	print("Total number of nodes pre-processing: {prev}".format(prev=prev_numNodes))
	print("Total number of nodes post-processing: {curr}".format(curr=curr_numNodes))
	
	print("------------------------------\n")


def get_child_dependence(node, mind, maxd):
	
	dependence = set()
	if len(node.children) > 0 and node.depth > mind and node.depth <= maxd:
		print("DD:", node.depth)
		find_all_dependence = [get_child_dependence(child, mind, maxd) for child in node.children]
		dependence = dependence.union(reduce(lambda x,y: x.union(y), find_all_dependence, set()))
		dependence.add(node)

	#return dependence.difference(common_dependence(node, mind, maxd))
	return dependence


def common_dependence(node, mind, maxd):
	find_all_dependence = [get_child_dependence(child, mind, maxd) for child in node.children]
	#print(type(node), [child.rec_eval(child) for child in node.children])
	#print([child for child in node.children])
	#print(find_all_dependence)
	if(len(find_all_dependence)!=0):
		common_subset = reduce(lambda x,y: x.intersection(y), find_all_dependence, find_all_dependence[0] )
	else:
		common_subset = set()
	return common_subset


def find_common_dependence( nodeList, mind, maxd ):
	
	common_subset = dict()

	for node in nodeList:
		preList = common_dependence(node, mind, maxd)
		redundant_list = reduce(lambda x,y: x.union(y), [common_dependence(n, mind, maxd) for n in preList], set())
		dep_set = preList.difference(redundant_list)
		common_subset[node] = set([node] if node.depth > mind and node.depth <= maxd else []) if len(dep_set)==0 else dep_set
#		common_subset[node] = common_dependence(node, mind, maxd)

	return common_subset


def get_opList(optype, maxd):
	allNodes = reduce(lambda x,y: x.union(y), [set(nodesList) for k,nodesList in Globals.depthTable.items() if k!=0 and k <= maxd], set())
	opList = set(filter(lambda x: x.token.type==optype, allNodes))
	return opList


def	parallelConcat(t1, t2):
	## atleast either t1 or t2 must be non-empry
	return t1 if len(t2)==0 else t2 if len(t1)==0 else t1+t2
	#return list(t1) if len(t2)==0 else list(t2) if len(t1)==0 else list(set(t1+t2))


##---------------------------------------
## First update each symtab internals with predicates
## Merge the symtabs for similar terms
## Lift the nodes with multiple options
##---------------------------------------

def parallel_merge(symTab1, symTab2, scope):

	assert(symTab1._symTab['_caller_'] == symTab2._symTab['_caller_'])

	_caller_ = symTab1._symTab['_caller_']
	f = lambda x,c : (x[0],x[1]&c)
	symTab1._symTab = {item : [f(x,symTab1._scopeCond) for x in symTab1._symTab[item]] for item in symTab1._symTab.keys() if item != '_caller_'}
	symTab2._symTab = {item : [f(x,symTab2._scopeCond) for x in symTab2._symTab[item]] for item in symTab2._symTab.keys() if item != '_caller_'}


	newtab = SymbolTable(scope, cond=Globals.__T__, \
	                    caller_symTab=_caller_)

	allkey = reduce(lambda x,y: x.union(y.keys()), \
	           [symTab1._symTab, symTab2._symTab], set())
	#allkey.remove('_caller_')

	g = lambda x, y: parallelConcat(x,y)
	newtab._symTab.update({k : g(symTab1._symTab.get(k,[]) , symTab2._symTab.get(k, [])) for k in allkey})

	## Now lift the nodes
	## it needs to be part of the parser then

	return newtab


def filterCandidate(bdmin, bdmax, dmax):
	#workList =  [[v[0] for v in nodeList if v[0].depth!=0]\
	#            for k,nodeList in Globals.GS[0]._symTab.items()]
	#workList =  [[v for v in nodeList if v.depth!=0]\
	workList =  [[v for v in nodeList if v.depth!=0 and v.depth>=bdmin and v.depth<=bdmax]\
	            for k,nodeList in Globals.depthTable.items()]
	
	workList = list(set(reduce(lambda x,y : x+y, workList, [])))
	#print("workList=",len(workList), [v.depth for v in workList])
	#print(bdmin, bdmax)
	
	return workList

	#return list(filter( lambda x:x.depth >= bdmin and x.depth <= bdmax ,\
	#							workList))
							   #[[v for v in nodeList if v.depth!=0] for k,nodeList in Globals.GS[0]._symTab.items()]
	                           #[v for k,v in Globals.GS[0]._symTab.items() if v.depth!=0]\
							 #))

def filterCandidate(bdmin, bdmax, dmax):
	
	workList = []
	opList = get_opList(DIV, bdmax)
	D = find_common_dependence(opList,5, bdmax)
	print(bdmin, bdmax, dmax, D)
	workList = list(reduce(lambda x,y : x.union(y), [v for k,v in D.items()], set()))
	if len(workList)==0:
		print("Empty WorkList!")
		pass
	else:
		maxdepth = max([n.depth for n in workList])
		print("1:From Filter Cands:", len(workList), len(opList))
		workList = [n for n in workList if n.depth == maxdepth]
		print("2:From Filter Cands:", len(workList), len(opList), [n.token.lineno for n in workList], maxdepth)

	for k, nodeList in Globals.depthTable.items():
		print("FC:", k, [n.depth for n in nodeList])
	if(len(workList) == 0):
		workList =  [[v for v in nodeList if v.depth!=0 and v.token.type in ops.DFOPS_LIST and v.depth>=bdmin and v.depth<=bdmax]\
	            for k,nodeList in Globals.depthTable.items()]

		print("WorkList:", [[type(n).__name__ for n in m] for m in workList])
		#workList = list(filter(lambda v : type(v).__name__ in ("TransOp", "BinOp", "Num", "LiftedOp"), workList))
		workList = list(set(reduce(lambda x,y : x+y, workList, [])))

	print("Final WorkList!", workList)
	return workList


def selectCandidateNodes(maxdepth, bound_mindepth, bound_maxdepth):
	
	PreCandidateList = filterCandidate(bound_mindepth, bound_maxdepth, maxdepth)

	loc_bdmax = bound_maxdepth
	while( len(PreCandidateList) <= 0 and loc_bdmax <= maxdepth):
		loc_bdmax += 5
		PreCandidateList = filterCandidate(bound_mindepth, loc_bdmax, maxdepth)

	#print(PreCandidateList)
	if(len(PreCandidateList) <= 0):
		return []
	else:
		f = lambda x : float(x.depth)/((loc_bdmax) + 0.01)
		g = lambda x, y : (-1)*y*math.log(y,2)*(len(x.parents)+ \
		                       (len(x.children) if type(x).__name__ == "LiftOp" else 0) +\
							   ops._Priority[x.token.type])
		##
		for cand in PreCandidateList:
			print("Else:", cand.token.type, cand.token.value, cand.token.lineno, cand.depth)
		##
		loc_bdmax = max([n.depth for n in PreCandidateList])
		cost_list = list(map( lambda x : [x.depth, g(x, f(x))], \
		                 PreCandidateList \
						))
		#print("bdmax:", loc_bdmax)
		print(cost_list)
		sum_depth_cost = [(depth, sum(list(map(lambda x:x[1] if x[0]==depth\
		                     else 0, cost_list)))) \
							 for depth in range(2, loc_bdmax+2)]
		print(sum_depth_cost)
		sum_depth_cost.sort(key=lambda x:(-x[1], x[0]))
		abs_depth = sum_depth_cost[0][0]


		## Obtain all candidate list at this level
		CandidateList = Globals.depthTable[abs_depth]

		print("CURRENT AST_DEPTH = : {ast_depth}".format(ast_depth=maxdepth))
		print("ABSTRACTION_DEPTH : {abs_depth}".format(abs_depth=abs_depth))

		return [abs_depth, CandidateList]


def writeToFile(results, fout, argList):

	inpfile = argList.file
	stdflag = argList.std
	sound = argList.sound

	fout.write("INPUT_FILE : "+inpfile+"\n")
	dumpStr = ''
	for outVar in Globals.outVars:
		#errIntv = results[Globals.lhstbl[outVar]]["ERR"]
		num_ulp_maxError = results[Globals.GS[0]._symTab[outVar][0][0]]["ERR"]
		num_ulp_SecondmaxError = results[Globals.GS[0]._symTab[outVar][0][0]]["SERR"]
		funcIntv = results[Globals.GS[0]._symTab[outVar][0][0]]["INTV"]

		#num_ulp_maxError = max([abs(i) for i in errIntv])
		maxError = num_ulp_maxError*pow(2, -53)
		SecondmaxError = num_ulp_SecondmaxError*pow(2, -53)
		print("FuncIntv:", funcIntv)
		outIntv = [funcIntv[0]-maxError-SecondmaxError, funcIntv[1]+maxError+SecondmaxError]
		abserror = (maxError + SecondmaxError)
		instability = results[Globals.GS[0]._symTab[outVar][0][0]]["INSTABILITY"] if argList.report_instability else "UNDEF"

		#print("//-------------------------------------")
		#print("Ouput Variable -> ", outVar)
		#print("Real Interval  -> ", funcIntv)
		#print("FP Interval    -> ", outIntv)
		#print("Absolute Error -> ", abserror)
		##print("Estimated bits preserved -> ", 52 - math.log(num_ulp_maxError,2))
		#print("//-------------------------------------\n\n")
		#print("Var:", outVar, "=>", results[Globals.lhstbl[outVar]])
		#print(k.f_expression, v)
		dumpStr += "\n//-------------------------------------\n"
		dumpStr += "VAR : "+ str(outVar) + "\n"
		dumpStr += "ABSOLUTE_ERROR : "+str(abserror)+"\n"
		dumpStr += "First-order Error : "+str(maxError)+"\n"
		if sound:
			dumpStr += "Higher-order Error : "+str(SecondmaxError)+"\n"
		dumpStr += "INSTABILITY : "+str(instability)+"\n"
		dumpStr += "REAL_INTERVAL : "+str(funcIntv)+"\n"
		dumpStr += "FP_INTERVAL : "+str(outIntv)+"\n"
		dumpStr += "//-------------------------------------\n"

	fout.write(dumpStr+"\n")
	if stdflag:
		print(dumpStr)


