

from gtokens import *
from lexer import Slex
from ASTtypes import *
from collections import deque
from SymbolTable import *

import copy
import sys
import helper

#import globals
import Globals
import time

import copy
from sympy.plotting.intervalmath import interval

class Sparser(object):

	tokens = Slex.tokens

	__slots__ = ['lexer', 'current_token', 'scopeStack', 'current_scope', 'current_symtab']

	def __init__(self, lexer):
		self.lexer = lexer
		self.current_token = None
		self.scopeStack = deque()
		self.current_symtab = None
		self.current_scope = Globals.scopeID


	def error(self):
		raise Exception('Invalid syntax while parsing')

	def consume(self, token_type):
		#print(self.current_token.type, token_type, self.current_token.value, self.current_token.lineno)
		if self.current_token.type == token_type:
			self.current_token = self.lexer.get_next_token()
		else:
			self.error()

	def addDepthInfo(self, node):
		Globals.depthTable[node.depth].add(node)

	def arith_expr(self):
		"""
		expr	:	term ( ( PLUS | MINUS) term )*
		"""

		node = self.arith_term()
		while self.current_token.type in (PLUS, MINUS):
			token = self.current_token
			self.consume(token.type)
			node = BinOp(left=node, token=token, right=self.arith_term())
			self.addDepthInfo(node)

		return node

	def arith_term(self):
		"""
		term	:	factor ( ( MUL | DIV) factor)*
		"""

		node = self.arith_factor()
		while self.current_token.type in (MUL, DIV):
			token = self.current_token
			self.consume(token.type)
			node = BinOp(left=node, token=token, right=self.arith_factor())
			self.addDepthInfo(node)

		return node

	def CheckSymTable(self, node, token):
		#symTable = Globals.GS[scopeID]
		lval = self.current_symtab.lookup(token.value)
		if lval is None or len(lval)==0:
			if token.value not in Globals.inputVars.keys():
				print(token.value)
				assert(token.value in Globals.inputVars.keys())
			self.current_symtab._symTab[token.value] = ((node, Globals.__T__),)
			#print("Check-tokens:", token, self.current_scope, lval, Globals.GS[0]._symTab.keys())
			return node
		elif (len(lval)==1):
			return lval[0][0]
		elif (len(lval) >= 2):
			#print(lval)
			#lval = lval+[[node,Globals.__T__]]
			newNode = LiftOp(lval, lval[0][0].token)
			self.addDepthInfo(newNode)
			self.current_symtab._symTab[token.value] = ((newNode,Globals.__T__),)
			return newNode

	def arith_factor(self):
		token = self.current_token
		if token.type in (INTEGER, FLOAT):
			self.consume(token.type)
			return Num(token)
		elif token.type in (SQRT, SIN, COS, ASIN, TAN, EXP, FPTRUNC, FPEXT):
			self.consume(token.type)
			node = TransOp(self.arith_factor(), token)
			self.addDepthInfo(node)
			return node
		elif token.type == LPAREN:
			self.consume(LPAREN)
			node = self.arith_expr()
			self.consume(RPAREN)
			self.addDepthInfo(node)
			return node
		else :
			node = Var(token)
			self.consume(ID)
			if(token.value == seng.var('px4')):
				print("MASSIVE DEBUG:", self.current_symtab.lookup(token.value))
			node = self.CheckSymTable(node, token)
			return node

	def get_new_condSym(self):
		Globals.CSID += 1
		return sym.var("CS"+str(Globals.CSID))

	def create_new_scope(self, adjacency='serial', cond=Globals.__T__):
		Globals.scopeID += 1
		print("Scope creattion: ID = ", Globals.scopeID)
		#item = None if len(self.scopeStack)<=0 else self.scopeStack.pop()
		if(adjacency == 'parallel'):
			_caller_ = self.current_symtab._symTab['_caller_']
			item = self.scopeStack.pop()
			item += (Globals.scopeID, )
		else:
			_caller_ = self.current_symtab
			item = (Globals.scopeID, )

		self.scopeStack.append(item)
		symTab = SymbolTable(Globals.scopeID, cond, _caller_)
		Globals.GS[Globals.scopeID] = symTab

		## update the current state
		self.current_symtab = symTab
		self.current_scope = Globals.scopeID

	##-------------------------------------------
	## The parsing of cond-expr and cond-term
	## are placeholders for now.
	## will be worked on details shortly
	##------------------------------------------- begin --
	def cond_expr(self):
		"""
		cond_expr	:	cond_term ( (&& | ||) cond_term)*
		"""
		node = self.cond_term()
		while self.current_token.type in (AND, OR):
			token = self.current_token
			self.consume(token.type)
			node = BinLiteral(left=node, token=token, right=self.cond_term())

		return node

	def cond_term(self):
		"""
		cond_term : ( expr inequalities expr )
		"""
		if (self.current_token.type == LPAREN):
			self.consume(LPAREN)
			node = self.cond_expr()
			self.consume(RPAREN)
			return node
		node = self.arith_expr()
		if self.current_token.type in (LT, LEQ, GT, GEQ, EQ, NEQ):
			token = self.current_token
			self.consume(token.type)
			node = ExprComp(left=node, token=token, right=self.arith_expr())
			#self.consume(RPAREN)
			return node
		else:
			self.error()
	##--------------------------------------------- end --


	def parallel_merge(self, symTab1, symTab2, scope):
		assert(symTab1._symTab['_caller_'] == symTab2._symTab['_caller_'])
		print("Stack:", self.scopeStack)
		_caller_ = symTab1._symTab['_caller_']
		## propagating predicate condition
		f = lambda x,c : (x[0],x[1]&c)
		symTab1._symTab = {item : tuple(f(x,symTab1._scopeCond) for x in symTab1._symTab[item])\
		                            for item in symTab1._symTab.keys() if item != '_caller_'}
		symTab2._symTab = {item : tuple(f(x,symTab2._scopeCond) for x in symTab2._symTab[item])\
		                            for item in symTab2._symTab.keys() if item != '_caller_'}
		newtab = SymbolTable(scope, cond=Globals.__T__, caller_symTab = _caller_)
		allkey = reduce(lambda x, y: x.union(y.keys()), [symTab1._symTab, symTab2._symTab], set())
		## merge common content from the two tabs
		g = lambda x, y : helper.parallelConcat(x, y)
		newtab._symTab.update({k : g(symTab1._symTab.get(k,tuple()) , symTab2._symTab.get(k, tuple())) for k in allkey})

		## now lift the nodes
		## lift only if more than 1 possibilites
		for k,v in newtab._symTab.items():
			if k != '_caller_' and len(v) > 1 :
				node = LiftOp(v, v[0][0].token)
				self.addDepthInfo(node)
				cond = reduce(lambda x,y: x|y, [x[1] for x in v])
				newtab._symTab[k] = ((node, cond),)
				#newtab._symTab[k] = [(node, Globals.__T__)]

		return newtab


	def serial_merge(self, symTab1, symTab2, scope):
		print( symTab1._scope, symTab2._scope)
		assert( symTab1._scope < symTab2._scope )
		assert( symTab2._symTab['_caller_'] == symTab1)
		_caller_ = symTab1._symTab['_caller_']
		## propagating predicate condition
		f = lambda x, c : (x[0], x[1]&c)
		symTab2._symTab = {item : tuple(f(x,symTab2._scopeCond) for x in symTab2._symTab[item])\
		                            for item in symTab2._symTab.keys() if item != '_caller_'}
		newtab = SymbolTable(scope, cond=symTab1._scopeCond, caller_symTab = _caller_)
		newtab._symTab = {k : v for k,v in symTab1._symTab.items()}

		for k,v in symTab2._symTab.items():
			if k == '_caller_':
				pass
			else:
				prev = symTab1.lookup(k)
				if prev is None:
					newtab._symTab[k] = v
				else:
					negCond = sympy.simplify(~(reduce(lambda x,y: x|y , [x[1] for x in v])))
					print(scope, k, negCond)
					newtab._symTab[k] = helper.parallelConcat(tuple((x[0],x[1]&negCond) for x in prev\
					                                     if(sym.simplify(x[1]&negCond)!=Globals.__F__)), \
				                                     symTab2._symTab[k])

			#elif k not in symTab1._symTab.keys():
			#	newtab._symTab[k] = v
			#else:
			#	negCond = sympy.simplify(~(reduce(lambda x,y: x|y , [x[1] for x in v])))
			#	print(scope, k, negCond)
			#	newtab._symTab[k] = helper.parallelConcat(tuple((x[0],x[1]&negCond) for x in symTab1._symTab[k]\
			#	                                     if(sym.simplify(x[1]&negCond)!=Globals.__F__)), \
			#	                                     symTab2._symTab[k])

		## check back if node lifting is required in serial merge
		return newtab




	def mergeSymTbale(self):
		print("Stack:", self.scopeStack)
		item = self.scopeStack.pop()
		symTab = Globals.GS[item[0]]
		if (len(item)>1):
			assert(len(item)==2)
			symTab = self.parallel_merge(Globals.GS[item[0]], Globals.GS[item[1]], scope=item[0])
		parent_scope_info = self.scopeStack.pop()
		if(len(parent_scope_info)>1): # parent 'else' is the immediate pred
			parent_item = parent_scope_info[1]
		else:
			parent_item = parent_scope_info[0]

		serialSymTab = self.serial_merge(Globals.GS[parent_item], symTab, parent_item)
		Globals.GS[parent_item] = serialSymTab
		self.current_scope = parent_item
		self.current_symtab = serialSymTab
		self.scopeStack.append(parent_scope_info)


	def ifblock(self):
		"""
		IFBLOCK : IF <cond_expr> then statements (empty | (else statements)) endif
		"""
		if self.current_token.type == IF:
			iftoken = self.current_token
			self.consume(IF)
			symCond = self.get_new_condSym()   ## the whole predicate identifier
			self.create_new_scope(adjacency='serial', cond=symCond) ## scope and symTab updated
			cond_node = self.cond_expr()
			Globals.predTable[symCond] = cond_node
			Globals.progTrace[self.current_token.lineno] = cond_node
			self.consume(THEN)
			true_node = self.statements()

			if(self.current_token.type==ELSE):
				self.consume(ELSE)
				self.create_new_scope(adjacency='parallel', cond=~symCond)
				false_node = self.statements()
			self.consume(ENDIF)
			
			self.mergeSymTbale()
			#item = self.scopeStack.pop()
		else:
			self.error()

	def assign_expr(self):
		if self.current_token.type == ID:
			name = str(self.current_token.value)
			nameSym = self.current_token.value
			nameToken = self.current_token
			self.consume(ID)
			rnd = 'rnd64'
			deriv_token = FLOAT
			if self.current_token.type == FPTYPE:
				rnd = str(self.current_token.value)
				deriv_token = FLOAT
				self.consume(FPTYPE)
			elif self.current_token.type == INTTYPE:
				rnd = str(self.current_token.value)
				deriv_token = INTEGER
				self.consume(INTTYPE)
			self.consume(ASSIGN)
			node = self.arith_expr()
			node.set_rounding(rnd)
			Globals.progTrace[self.current_token.lineno] = node  # prog Trace for the assign expr
			node.derived_token = deriv_token
			self.consume(SEMICOLON)
			self.current_symtab._symTab[nameSym] = ((node, Globals.__T__),)
			return node

			



	def statements(self):
		"""
		statements	:	<IFBLOCK>		<statements>
					|	<ASSIGNEXPR>	<statements>
					|	<empty>
		"""

		node = EmptyNode()
		while self.current_token.type in (IF, ID):
			if self.current_token.type == IF:
				node = self.ifblock()
			elif self.current_token.type == ID:
				node = self.assign_expr()
		return node


	def program(self):
		""" program : INPUTS OUTPUTS EXPRS """
		print("Inside Program")
		## Create symbolTable in the Global scope right here
		self.create_new_scope(adjacency='serial')
		##
		self.parse_input()
		self.parse_output()
		self.parse_constraints()
		self.parse_exec()

	def parse_constraints(self):
		if self.current_token.type != REQUIRES:
			pass
		else:
			self.consume(REQUIRES)
			self.consume(SLPAREN)
			self.constraint_list()
			self.consume(SRPAREN)

	def constraint_list(self):
		""" constraint_list : cond_expr | ; | empty """
		self.assign_constraint_expr()

		while self.current_token.type == SEMICOLON:
			self.consume(SEMICOLON)
			self.assign_constraint_expr()

	def assign_constraint_expr(self):
		print(""" assign_constraint_expr : ID COLON cond_expr """)
		if self.current_token.type == ID:
			name = str(self.current_token.value)
			nameToken = self.current_token
			self.consume(ID)
			self.consume(COLON)
			node = self.cond_expr()
			print("CONSTRAINTS: ", node.rec_eval(node))
			node_exists = Globals.externPredTable.get(nameToken.value, None)
			if node_exists is not None:
				self.error()
			Globals.externPredTable[nameToken.value] = node
		
		


	def parse_exec(self):
		self.consume(EXPRS)
		self.consume(SLPAREN)
		begin_node = self.statements()
		self.consume(SRPAREN)
		## Do a final lifting of any remaining gateaways
		assert(self.current_scope == 0)
		assert(self.current_symtab == Globals.GS[0])
		del self.current_symtab._symTab['_caller_']
		for k,v in self.current_symtab._symTab.items():
			if len(v) > 1 :
				#print(k)
				#assert(k in Globals.outVars)
				node = LiftOp(v, v[0][0].token)
				self.addDepthInfo(node)
				cond = reduce(lambda x,y: x|y, [x[1] for x in v])
				self.current_symtab._symTab[k] = ((node, cond),)
				#newtab._symTab[k] = [(node, Globals.__T__)]


	def parse_output(self):
		""" outputs { outputs list } """
		self.consume(OUTPUTS)
		self.consume(SLPAREN)
		self.output_list()
		self.consume(SRPAREN)

	def output_list(self):
		""" output_list : output | ; | empty """
		self.output()

		while self.current_token.type == SEMICOLON:
			self.consume(SEMICOLON)
			self.output()

	def output(self):
		while self.current_token.type == ID:
			nameSym = self.current_token.value
			self.consume(ID)
			Globals.outVars.append(nameSym)


	def parse_input(self):
		""" input { interval_list } """
		self.consume(INPUTS)
		self.consume(SLPAREN)
		self.interval_list()
		self.consume(SRPAREN)

	def interval_list(self):
		""" interval_list : interval | ; | empty """
		self.interval()

		while self.current_token.type == SEMICOLON:
			self.consume(SEMICOLON)
			self.interval()

	##---------------------------------------------------------------------

	def intv_expr(self):
		"""
		expr	:	term ( ( PLUS | MINUS) term )*
		"""

		node = self.intv_term()
		while self.current_token.type in (PLUS, MINUS):
			token = self.current_token
			self.consume(token.type)
			node = BinOp(left=node, token=token, right=self.intv_term())

		return node

	def intv_term(self):
		"""
		term	:	factor ( ( MUL | DIV) factor)*
		"""

		node = self.intv_factor()
		while self.current_token.type in (MUL, DIV):
			token = self.current_token
			self.consume(token.type)
			node = BinOp(left=node, token=token, right=self.intv_factor())

		return node


	def intv_factor(self):
		token = self.current_token
		if token.type in (INTEGER, FLOAT):
			self.consume(token.type)
			return Num(token)
		elif token.type in (SQRT, SIN, COS, ASIN, TAN, EXP):
			self.consume(token.type)
			node = TransOp(self.intv_factor(), token)
			return node
		elif token.type == LPAREN:
			self.consume(LPAREN)
			node = self.intv_expr()
			self.consume(RPAREN)
			return node
		elif token.type == ID:
			node = FreeVar(token)
			dep_var = Globals.inputVars.get(token.value)
			self.consume(ID)
			if dep_var is None:
				print("Undefined Variable in input declaration expression")
				self.error()
			else:
				return node
		else :
			self.error()


	##---------------------------------------------------------------------

	def intv_eval(self, expr, lower=False):
		free_syms = list(expr.free_symbols)
		if len(free_syms)==1:
			fsym = free_syms[0]
			f = sympy.lambdify(fsym, expr)
			intv = f ( interval(Globals.inputVars[fsym]["INTV"][0], Globals.inputVars[fsym]["INTV"][1]) )
		else:
			f = sympy.lambdify([tuple(free_syms)], expr)
			intv = f( [ tuple( interval(Globals.inputVars[var]["INTV"][0], Globals.inputVars[var]["INTV"][1])   for var in free_syms ) ] )
	
		return intv.mid-intv.width/2 if lower else intv.mid+intv.width/2



	def interval(self):
		while self.current_token.type == ID:
			var_token = self.current_token
			name = str(self.current_token.value)
			self.consume(ID)
			fptype = str(self.current_token.value)
			self.consume(FPTYPE)
			self.consume(COLON)
			self.consume(LPAREN)
			## check bater later here for expressions
			n = self.intv_expr()
			left = n.rec_eval(n)
			lexpr = left[0].exprCond[0]
			lexpr = lexpr if seng.count_ops(lexpr)==0 else self.intv_eval(lexpr, lower=True)
			#left = self.current_token.value
			#self.consume(FLOAT)
			self.consume(COMMA)

			n = self.intv_expr()
			right = n.rec_eval(n)
			rexpr = right[0].exprCond[0]
			rexpr = rexpr if seng.count_ops(rexpr)==0 else self.intv_eval(rexpr)

			#print("Interval:", lexpr, rexpr)
			#self.consume(COMMA)
			#right = self.current_token.value
			#self.consume(FLOAT)
			self.consume(RPAREN)

			symVar = FreeVar(var_token, cond=Globals.__T__)
			symVar.set_noise(symVar, (rexpr*pow(2,-53), 0.0))
			symVar.set_rounding(fptype)
			#self.current_symtab.insert(var_token.value, [[symVar,Globals.__T__]])
			self.current_symtab.insert(var_token.value, ((symVar,Globals.__T__),)   )
			Globals.inputVars[var_token.value] = {"INTV" : [lexpr, \
															rexpr]}

	def parse(self, text):
		self.lexer.create_token_generator(text)
		self.current_token = self.lexer.get_next_token()#current_token()
		self.program()
		#print(self.current_symtab._symTab.keys())
		#print(self.current_symtab._scope)
		#print(self.current_symtab._scopeCond)




if __name__ == "__main__":
	
	start_parse_time = time.time()
	text = open(sys.argv[1], 'r').read()
	lexer = Slex()
	parser = Sparser(lexer)
	parser.parse(text)
	end_parse_time = time.time()
	print("Parsing time -> ", end_parse_time - start_parse_time)

	sumNodes = 0
	for d,num in Globals.depthTable.items():
		print(d, "-->", len(num))
		sumNodes += len(num)
	print("Total = ", sumNodes)
	print("NumPredicates = ", len(Globals.predTable.keys()))
	
#	for k,v in Globals.progTrace.items():
#		print("Line {num} , val = {expr}".format(num=k, expr=v.rec_eval(v)))

#	for k,v in Globals.predTable.items():
#		print(k, v.token.lineno, v.rec_eval(v))
#		#if "True" in v.rec_eval(v):
#		#	print(k, v.token.lineno, v.rec_eval(v))
#		
#
	for k,v in Globals.GS[0]._symTab.items():
		print(k, v)
		for els in v:
			(nd, cond) = els
			for child in nd.nodeList:
				print(child[0], child[1])

	#rhs = Globals.GS[0]._symTab[seng.var('g')]
	#print(rhs[0][0], type(rhs[0][0]))
	#print(rhs[0][0].rec_eval(rhs[0][0]))
#
#	for k,v in Globals.depthTable.items():
#		print(k, len(v))
