
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


# Builds expression for each node in the "node" subtree of the AST.
def build_expression_at_node(node, reachable, parent_dict, free_syms, cond_syms, cond, etype, ctype, inv=False):
	"""
	Builds expression for each node in the "node" subtree of the AST.
	Also builds the parent_dict, free_syms, cond_syms sets and reachable dict.

	Parameters
	----------
	node : node type
		Any node.
	reachable : defaultdict(set)
		Dictionary of depth(int) -> set(nodes)
	parent_dict : defaultdict(list)
		Dictionary of node -> [parents]
	free_syms : set
		Accumulated Set of symbolic variables from all ExprComp nodes.
	cond_syms : set
		Accumulated Set of symbolic variables from all Conditionals in Sym structures.
	etype : bool
		Flag to include error in conditional when generating expressions for conditionals
	ctype : bool
		Flag to retrieve conditional and free symbols when handling conditionals. We get free symbols out of this function
		if this flag is set.
	inv : ?

	Returns
	-------
	free_syms : set
		Accumulated Set of symbolic variables from all ExprComp nodes.
	cond_syms : set
		Accumulated Set of symbolic variables from all Conditionals in Sym structures.
	"""
	# Recursive call to build expressions for the subtrees of "node"
	for child in node.children:
		if not reachable[child.depth].__contains__(child):
			(free_syms, cond_syms) = build_expression_at_node(child, reachable, parent_dict, free_syms, cond_syms, cond, etype, ctype, inv)

		parent_dict[child].append(node)

	# Expression building for ExprComp nodes
	if type(node).__name__ == "ExprComp":
		print("ExprComp line:", node.token.lineno)

		# etype flag determines whether to include error when building expressions for conditionals.
		if etype:
			print("HIDDEN CONDITIONAL DEPTH:", node.children[0].depth, node.children[1].depth)
			res0 = ANC([node.children[0]], [], node.children[0].depth, Globals.argList.use_atomic_conditions, Globals.argList.realpaver).start()
			res1 = ANC([node.children[1]], [], node.children[1].depth, Globals.argList.use_atomic_conditions, Globals.argList.realpaver).start()

			# An ExprComp node has modified evaluation function to include the extra error terms
			# The ExprComp node adds the above calculated concretized error to the conditional expression.
			(fexpr,fsyms) = node.mod_eval(node, inv, res0[node.children[0]]["ERR"]*pow(2,-53), \
								   res1[node.children[1]]["ERR"]*pow(2,-53) )

			# TODO: What does this print statement mean? Why did Arnab say this?
			print("Never ever error")
		else:
			(fexpr,fsyms) = node.mod_eval(node, inv, 0.0, 0.0)
		free_syms = free_syms.union(fsyms)
	# Expression building for nodes other than ExprComp nodes
	else:
		fexpr = node.eval(node, inv)

	node.set_expression(fexpr)
	#print("FEXPRESSION TYPE = ", type(node.f_expression).__name__)
	# Symbols in the predicate part of SymTup are added to cond_syms set.
	# New symbols are added on encountering a new predicate on entering an "if then else" block.
	if type(node.f_expression).__name__ == "SymTup":
		csymSet = reduce(lambda x,y: x.union(y), \
						[el.exprCond[1].free_symbols for el in node.f_expression if el.exprCond[1] not in (True, False)],
						 set())
		cond_syms = cond_syms.union(csymSet)

	# Adding node to the reachable dictionary.
	reachable[node.depth].add(node)

	return free_syms, cond_syms


	#print(node.depth, type(node).__name__, node.cond)
	#print("main:", type(node).__name__, fexpr)
	#print(node.token)
	#print([(type(child).__name__, child.f_expression) for child in node.children], "\n\n")
	#fexpr = node.eval(node)
	#node.set_expression(fexpr)
	#reachable[node.depth].add(node)


# Starts building expressions for all nodes in candidate_list
def expression_builder_driver(candidate_list, etype=False, ctype=False, inv=False):
	"""
	Driver function for building expressions at all nodes in subtree starting with given root nodes in candidate_list.

	Parameters
	----------
	candidate_list : list
		List of nodes in the AST representing some expressions.
	# TODO: See if etype parameter is needed at all. Why is it needed in expression building?
	etype : bool
		Flag to include error in conditional when generating expressions for conditionals
	ctype : bool
		Flag to retrieve conditional and free symbols when handling conditionals. We get free symbols out of this function
		if this flag is set.
	inv : ?

	Returns
	-------
	free_syms : set
		Accumulated Set of symbolic variables from all ExprComp nodes.
	cond_syms : set
		Accumulated Set of symbolic variables from all Conditionals in Sym structures.
	parent_dict : defaultdict(list)
		Dictionary of node -> [parents]
	"""

	parent_dict = defaultdict(list)
	reachable = defaultdict(set)
	free_syms = set()
	cond_syms = set()
	if ctype:
		print("Beginning building expressions for conditionals...")
	else:
		print("Beginning building expressions...")

	# For each node, build the complete expression within the node itself and accumulate free_syms and cond_syms
	for node in candidate_list:
		if not reachable[node.depth].__contains__(node):
			# print(node.depth)
			(free_syms, cond_syms) = build_expression_at_node(node, reachable, parent_dict, free_syms, cond_syms, cond=Globals.__T__,etype=etype, ctype=ctype, inv=inv)

		#print(node.f_expression)
		#print("From expression builder: Root node stats -> opcount={opcount}, depth={depth}".format(opcount=0 if type(node).__name__ not in ('TransOp', 'BinOp') else node.f_expression.__countops__(), depth=node.depth))
		#print(type(node).__name__, node.token.type, node.depth, node.f_expression)
		#print([(type(child).__name__, child.token.type, child.depth, child.f_expression) for child in node.children])

	del reachable

	if ctype:
		print("Completed building expressions for conditionals...")
	else:
		print("Completed building expressions...")
	if ctype:
		return free_syms, cond_syms
	else:
		return parent_dict, cond_syms


# TODO: Figure out what inv is from the definition Arnab wrote here.
# Builds a conditional string from all nodes in conditional_node_list.
# inv  = To generate delta inverse that includes the grey-zone
def handleConditionals(conditional_node_list, etype=True, inv=False):
	"""
	Generates expressions corresponding to the conditional nodes in conditional_node_list.
	Builds a conditional string from the generated expressions.

	Parameters
	----------
	conditional_node_list : list
		List of conditional nodes. Conditional nodes can be external constraints or if-then-else conditionals.
	etype : bool
		Flag to include error in conditional when generating expressions for conditionals
	inv : ?

	Returns
	-------
	cstr : str
		Conditional string to be used for the optimizer.
	fsyms : set
		Accumulated Set of symbolic variables from all ExprComp nodes.
	csyms : set
		Accumulated Set of symbolic variables from all Conditionals in Sym structures.
	"""
	print("Building conditional expressions...\n")
	logger.info("Building conditional expressions...\n")
	(fsyms, csyms) = expression_builder_driver(conditional_node_list, etype, ctype=True, inv=inv)

	# Builds a
	cstr = " & ".join([str(conditional_node.f_expression) for conditional_node in conditional_node_list])
	print("Debug-check:", (cstr, fsyms, csyms))
	#return (" & ".join([str(conditional_node.f_expression) for conditional_node in conditional_node_list]),fsyms, csyms)
	return cstr, fsyms, csyms

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

	candidate_list = [Globals.global_symbol_table[0]._symTab[outVar] for outVar in Globals.outVars]
	reachable = defaultdict(set)


	for nodeList in candidate_list:
		assert(len(nodeList)==1)
		[node,cond] = nodeList[0]
		if not reachable[node.depth].__contains__(node):
			pretraverse(node, reachable)

	print("Symbol count Pre Processing :", len(Globals.global_symbol_table[0]._symTab.keys()))
	Globals.global_symbol_table[0]._symTab = {syms: tuple(set(n for n in nodeCondList \
										if reachable[n[0].depth].__contains__(n[0]))) \
										for syms,nodeCondList in Globals.global_symbol_table[0]._symTab.items() }
	print("Symbol count Post Processing :", len(Globals.global_symbol_table[0]._symTab.keys()))
	prev_numNodes = sum([ len(Globals.depthTable[el]) for el in Globals.depthTable.keys() if el!=0] )
	Globals.depthTable = reachable
	curr_numNodes = sum([ len(Globals.depthTable[el]) for el in Globals.depthTable.keys() if el!=0] )
	logger.info("Total number of nodes pre-processing: {prev}".format(prev=prev_numNodes))
	logger.info("Total number of nodes post-processing: {curr}".format(curr=curr_numNodes))
	print("Total number of nodes pre-processing: {prev}".format(prev=prev_numNodes))
	print("Total number of nodes post-processing: {curr}".format(curr=curr_numNodes))

	print("------------------------------\n")


def get_nodes_within_bounds(node, minimum_depth, maximum_depth):
	"""
	Creates a flattened subset of nodes below 'node' that lie between minimum_depth and maximum_depth

	Parameters
	----------
	node : node type
		Any node
	minimum_depth : int
		Lower depth used to filter nodes
	maximum_depth : int
		Upper depth used to filter nodes

	Returns
	-------
	dependent_node_set : set
		Set of nodes that 'node' depends on within specified depth bounds
	"""
	dependent_node_set = set()

	if len(node.children) > 0 and minimum_depth < node.depth <= maximum_depth:
		for child in node.children:
			dependent_node_set = dependent_node_set.union(get_nodes_within_bounds(child, minimum_depth, maximum_depth))

	return dependent_node_set


def common_nodes(node, minimum_depth, maximum_depth):
	"""
	Finds nodes that are common between the flattened set of nodes of the children of 'node' within minimum_depth and
	maximum_depth

	Parameters
	----------
	node : node type
		Any node
	minimum_depth : int
		Lower depth used to filter nodes
	maximum_depth : int
		Upper depth used to filter nodes

	Returns
	-------
	common_nodes_set : set
		Set of nodes common between the nodes below the children of 'node' lyging between the specified bounds.
	"""
	print("Getting all nodes within ", minimum_depth, " < depth <= ", maximum_depth, "below node: ", node)
	dependent_node_list = [get_nodes_within_bounds(child, minimum_depth, maximum_depth) for child in node.children]
	print(dependent_node_list)

	# Appending node as it would not be in the above generated list of sets
	if node not in dependent_node_list:
		dependent_node_list.append({node})

	# Finding common nodes between the node sets of children
	if len(dependent_node_list)!=0:
		common_nodes_set = reduce(lambda x,y: x.intersection(y), dependent_node_list, dependent_node_list[0])
	else:
		common_nodes_set = set()

	return common_nodes_set


def find_common_dependence(node_list, minimum_depth, maximum_depth):
	"""
	In a list, for each node, finds set of nodes below its children that are common and removes redundant nodes.

	Parameters
	node_list : list
		List of nodes
	minimum_depth : int
		Lower depth used to filter nodes
	maximum_depth : int
		Upper depth used to filter nodes

	Returns
	-------
	common_node_dict : dict
		Dictionary[node -> set] of nodes that 'node' depends on
	"""
	common_nodes_dict = dict()

	for node in node_list:
		# Finding nodes common between the nodes below children of 'node'
		initial_dependence_list = common_nodes(node, minimum_depth, maximum_depth)

		# Finding nodes that are common between nodes below nodes from the above list
		redundant_nodes_list = reduce(lambda x, y: x.union(y), [common_nodes(n, minimum_depth, maximum_depth) for n in
																initial_dependence_list], set())

		# Removing redundant nodes found from initial list and adding to dictionary
		final_dependence_set = initial_dependence_list.difference(redundant_nodes_list)
		common_nodes_dict[node] = set([node] if minimum_depth < node.depth <= maximum_depth
									  else []) if len(final_dependence_set)==0 else final_dependence_set

	return common_nodes_dict


def filter_nodes_with_operation_within_depth(operation_token_type, max_depth):
	"""
	Creates a set of all nodes within depth max_depth from all nodes in depthTable and from it filters nodes with token
	type operation_token_type.

	Parameters
	----------
	operation_token_type : const
		Type of token to filter out nodes
	max_depth : int
		Max depth for filtration process

	Returns
	-------
	node_set : set
		Filtered nodes
	"""
	node_set = reduce(lambda x,y: x.union(y), [set(nodesList) for k,nodesList in Globals.depthTable.items() if k != 0 and
											   k <= max_depth], set())
	filtered_node_list = set(filter(lambda x: x.token.type == operation_token_type, node_set))

	return filtered_node_list


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

# def filter_candidates_for_abstraction(bdmin, bdmax, dmax):
# 	#workList =  [[v[0] for v in node_set if v[0].depth!=0]\
# 	#            for k,node_set in Globals.global_symbol_table[0]._symTab.items()]
# 	#workList =  [[v for v in node_set if v.depth!=0]\
# 	workList =  [[v for v in node_set if v.depth!=0 and v.depth>=bdmin and v.depth<=bdmax]\
# 	            for k,node_set in Globals.depthTable.items()]
#
# 	workList = list(set(reduce(lambda x,y : x+y, workList, [])))
# 	#print("workList=",len(workList), [v.depth for v in workList])
# 	#print(bdmin, bdmax)
#
# 	return workList
#
# 	#return list(filter( lambda x:x.depth >= bdmin and x.depth <= bdmax ,\
# 	#							workList))
# 							   #[[v for v in node_set if v.depth!=0] for k,node_set in Globals.global_symbol_table[0]._symTab.items()]
# 	                           #[v for k,v in Globals.global_symbol_table[0]._symTab.items() if v.depth!=0]\
# 							 #))


def filter_candidates_for_abstraction(lower_bound, upper_bound, maximum_depth):
	"""
	Filters nodes with depth within lower_bound and upper

	Parameters
	----------
	lower_bound : int
		The lower bound of the filter.
	upper_bound : int
		The upper bound of the filter.
	maximum_depth : int
		The max depth of the ast

	Returns
	-------
	dependence_node_list : list
		List of filtered nodes within the specified depth bounds
	"""
	filtered_node_set = filter_nodes_with_operation_within_depth(DIV, upper_bound)
	common_dependence_node_dict = find_common_dependence(filtered_node_set, 5, upper_bound)
	print(lower_bound, upper_bound, maximum_depth, common_dependence_node_dict)
	dependence_node_list = list(reduce(lambda x,y : x.union(y), [v for k,v in common_dependence_node_dict.items()], set()))
	if len(dependence_node_list)==0:
		print("Empty dependence_node_list! Generating candidates")
		dependence_node_list = [[v for v in node_list if
								 v.depth != 0 and v.token.type in ops.DFOPS_LIST and v.depth >= lower_bound and v.depth <= upper_bound] \
								for k, node_list in Globals.depthTable.items()]

		print("dependence_node_list:", [[type(n).__name__ for n in m] for m in dependence_node_list])
		# dependence_node_list = list(filter(lambda v : type(v).__name__ in ("TransOp", "BinOp", "Num", "LiftedOp"), dependence_node_list))
		dependence_node_list = list(set(reduce(lambda x, y: x + y, dependence_node_list, [])))
		pass
	else:
		# Find greatest depth from all nodes in the dependence_node_list
		maxdepth = max([n.depth for n in dependence_node_list])
		print("1:From Filter Cands:", len(dependence_node_list), len(filtered_node_set))

		# Select nodes with depth=maxdepth
		dependence_node_list = [n for n in dependence_node_list if n.depth == maxdepth]
		print("2:From Filter Cands:", len(dependence_node_list), len(filtered_node_set), [n.token.lineno for n in dependence_node_list], maxdepth)

	# for k, node_list in Globals.depthTable.items():
	# 	print("FC:", k, [n.depth for n in node_list])

	print("Final dependence_node_list!", dependence_node_list)
	return dependence_node_list


def select_abstraction_candidate_nodes(max_depth, lower_bound_depth, upper_bound_depth):
	"""
	Selects the nodes for abstraction within specified depth window
	Criteria of node selection:
	1) Needs to be within depth window.
	
	Parameters
	----------
	max_depth : int
		Depth of the AST
	lower_bound_depth : int
		Lower bound of depth window for filtering nodes for abstraction
	upper_bound_depth : int
		Upper bound of depth window for filtering nodes for abstraction
	Returns
	-------
	(int, list)
		Depth of nodes to be abstracted and Candidate nodes chosen for abstraction
	"""
	initial_candidate_list = filter_candidates_for_abstraction(lower_bound_depth, upper_bound_depth, max_depth)

	local_upper_bound = upper_bound_depth
	# Keeps increasing upper bound of depth window till we get some candidate nodes for abstraction
	while( len(initial_candidate_list) <= 0 and local_upper_bound <= max_depth):
		local_upper_bound += 5
		initial_candidate_list = filter_candidates_for_abstraction(lower_bound_depth, local_upper_bound, max_depth)

	#print(initial_candidate_list)
	if(len(initial_candidate_list) <= 0):
		# Return since no candidates found for abstraction
		return []
	else:
		f = lambda x : float(x.depth)/((local_upper_bound) + 0.01)
		g = lambda x, y : (-1)*y*math.log(y,2)*(len(x.parents)+ \
		                       (len(x.children) if type(x).__name__ == "LiftOp" else 0) +\
							   ops._Priority[x.token.type])
		##
		for cand in initial_candidate_list:
			print("Else:", cand.token.type, cand.token.value, cand.token.lineno, cand.depth)
		##
		local_upper_bound = max([n.depth for n in initial_candidate_list])
		cost_list = list(map( lambda x : [x.depth, g(x, f(x))], \
		                 initial_candidate_list \
						))
		#print("bdmax:", local_upper_bound)
		print(cost_list)
		sum_depth_cost = [(depth, sum(list(map(lambda x:x[1] if x[0]==depth\
		                     else 0, cost_list)))) \
							 for depth in range(2, local_upper_bound+2)]
		print(sum_depth_cost)
		sum_depth_cost.sort(key=lambda x:(-x[1], x[0]))
		abstraction_depth = sum_depth_cost[0][0]


		## Obtain all candidate list at this level
		candidate_list = Globals.depthTable[abstraction_depth]

		print("CURRENT AST_DEPTH = : {ast_depth}".format(ast_depth=max_depth))
		print("ABSTRACTION_DEPTH : {abstraction_depth}".format(abstraction_depth=abstraction_depth))

		return [abstraction_depth, candidate_list]


def writeToFile(results, fout, argList):

	inpfile = argList.file
	stdflag = argList.std
	sound = argList.sound

	fout.write("INPUT_FILE : "+inpfile+"\n")
	dumpStr = ''
	for outVar in Globals.outVars:
		#errIntv = results[Globals.lhstbl[outVar]]["ERR"]
		num_ulp_maxError = results[Globals.global_symbol_table[0]._symTab[outVar][0][0]]["ERR"]
		num_ulp_SecondmaxError = results[Globals.global_symbol_table[0]._symTab[outVar][0][0]]["SERR"]
		funcIntv = results[Globals.global_symbol_table[0]._symTab[outVar][0][0]]["INTV"]

		#num_ulp_maxError = max([abs(i) for i in errIntv])
		maxError = num_ulp_maxError*pow(2, -53)
		SecondmaxError = num_ulp_SecondmaxError*pow(2, -53)
		print("FuncIntv:", funcIntv)
		outIntv = [funcIntv[0]-maxError-SecondmaxError, funcIntv[1]+maxError+SecondmaxError]
		abserror = (maxError + SecondmaxError)
		instability = results[Globals.global_symbol_table[0]._symTab[outVar][0][0]]["INSTABILITY"] if argList.report_instability else "UNDEF"

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


