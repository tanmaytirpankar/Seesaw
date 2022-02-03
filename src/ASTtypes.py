import Globals 

import sympy as sym
import ops_def as ops

from gtokens import *

from functools import reduce
from PredicatedSymbol import Sym, SymTup, SymConcat
import numpy as np


class CToken(object):
	"""
	Representation of a single token.

	Attributes
	----------
	type : string
		Represents the type of token.
	value : None
	lineno : int
		Never used.
	derived_token : string
		Representing numerical type using token values
	"""
	__slots__ = ('type', 'value', 'lineno', 'derived_token')
	def __init__(self, tp, value):
		self.type = tp
		self.value = value
		self.lineno = None
		self.derived_token = FLOAT

	def __str__(self):
		return '(Token.type = {ttype}, Token.value={value}, Token.lineno=Lifted(cannot trace origin))'.format(ttype=self.type, value=self.value)

	def __repr__(self):
		return self.__str__()


##-- Base AST class
class AST(object):
	# TODO: What is the difference between children and nodeList?
	"""
	The AST node: base class of all nodes.

	Attributes
	----------
	depth : int
		The depth of the node from the root. The default is 0.
	f_expression : string
		The expression represented by the node. Default is none.
	children : tuple of nodes
		The children of the node. Can have at most 2 children. The default is an empty tuple.
	parents : set of nodes
		The parent set of the node. Root node has no parent. The default is an empty set.
	noise : tuple of noise interval.
		Interval representing noise generated by abstracted node. The default is (0,0).
	rnd : float
		The rounding value at the node. The default is 1.0 implying no rounding necessary.
	cond : Boolean.
		The boolean value of the conditional node indicating whether it is active (Contributing to the final error)
		The default is True.
	nodeList	:	List
		The list of nodes.
	derived_token : string
		Representing numerical type using token values.

	Methods
	-------
	set_expression(fexpr)
		Sets the f_expression class attribute.
	"""
	__slots__ = ['depth', 'f_expression', 'children', 'parents', 'noise', 'rnd', 'cond', 'nodeList', 'derived_token']

	def __init__(self, cond=Globals.__T__):
		self.depth = 0
		self.f_expression = None
		self.children = ()
		self.parents = ()
		self.noise = (0, 0)
		self.rnd = 1.0
		self.cond = cond
		self.nodeList = []
		self.derived_token = FLOAT

	def __repr__(self):
		"""
		Returns
		-------
		repr_str : str
			String representation of the class.
		"""
		parent_str = "("
		for i in range(len(self.parents)):
			parent_str = parent_str + self.parents[i].token.value + ","
		parent_str = parent_str + ")"
		repr_str = '\n\tdepth:' + repr(self.depth) \
			+ '\n\tparents:' + parent_str \
			+ '\n\tnoise:' + repr(self.noise) \
			+ '\n\trnd:' + repr(self.rnd)
		return repr_str

	def set_expression(self, fexpr):
		"""
		Sets the f_expression class attribute.

		Parameters
		----------
		fexpr : string
			The expression represented by the node.
		"""
		if isinstance(fexpr, SymTup) or isinstance(fexpr, str):
			self.f_expression = fexpr
		else:
			print("fexpression is not of type SymTup")
			exit(1)

	def eval(self, inv=False):
		"""
		Returns the numerical value stored by the token.
		Parameters
		----------
		inv : ?
		Returns
		-------
		self.f_expression : int
			The value as parsed by the parser.
		"""
		return self.f_expression

	@staticmethod
	def get_noise(obj):
		return obj.f_expression if obj.f_expression is not None else 0.0

	@staticmethod
	def rec_eval(obj):
		"""
		Returns the expression represented by the node after expanding out.
		Parameters
		----------
		obj : node type
		Returns
		-------
		obj.eval(obj) : string
		"""
		return obj.eval(obj)

	def set_rounding(self, rnd_type):
		self.rnd = ops._FP_RND[rnd_type]

	def get_rounding(self):
		return self.rnd * 1.0






##-- EmptyNode
class EmptyNode(AST):
	pass





##-- Numeric Node	
class Num(AST):
	"""
	The Num node: Derived from AST representing numerical value holding nodes.
	Attributes
	----------
	token : Lexer token object (INTEGER, FLOAT tokens)
	"""
	__slots__ = ['token']

	def __init__(self, token, cond=Globals.__T__):
		super().__init__()
		self.token = token
		self.f_expression = self.eval(self)
		self.rnd = 0.0
		self.cond = cond
		self.derived_token = token.type

	def __repr__(self):
		"""
		Returns
		-------
		repr_str : str
			String representation of the class.
		"""
		repr_str = '\nNum{' + '\n\ttoken:' + repr(self.token) + super().__repr__() + '\n}'
		return repr_str

	@staticmethod
	def eval(obj, inv=False):
		# TODO: Find what inv is?
		"""
		Returns the numerical value stored by the token.

		Parameters
		----------
		inv : ?

		Returns
		-------
		SymTup : Object
			An object of class SymTup containing the value as parsed by the parser and True boolean value for the
			conditional node.
		"""
		return  SymTup((Sym(obj.token.value, Globals.__T__),))

	@staticmethod
	def get_noise(obj):
		"""
		# TODO: High Precision and Low Precision numbers here are fixed to be double and single precision. Need change?
		Calculates the error in the number using the difference between a high and low precision value.
		Parameters
		----------
		obj : node type
		Returns
		-------
		Nothing
		"""
		#return abs(float(BigFloat(obj.token.value,context=single_precision) - BigFloat(obj.token.value,context=double_precision)))
		return np.float64(obj.token.value) - np.float32(obj.token.value)
		#return obj.token.value*pow(2,-53)
		#return 0.0





##-- FreeVariable
class FreeVar(AST):
	"""
	The FreeVar node: Derived from AST representing the input variables in the section INPUTS.
	Attributes
	----------
	token : Lexer token object (ID token in the EXPRS section that are not reserved keywords)
	"""
	__slots__ = ['token']
	def __init__(self, token, cond=Globals.__T__):
		super().__init__()
		self.token = token
		self.cond = cond
		self.derived_token = FLOAT ;

	def __repr__(self):
		"""
		Returns
		-------
		repr_str : str
			String representation of the class.
		"""
		repr_str = '\nFreeVar{' + '\n\ttoken:' + repr(self.token) + super().__repr__() + '\n}'
		return repr_str

	@staticmethod
	def eval(obj, round_mode="fl64", inv=False):
		"""
		Sets rounding and returns the interval node or a new symengine variable.
		Parameters
		----------
		self : node type
		round_mode : string
			Describes rounding mode at the node.
		Returns
		-------
		intv["INTV"][0] : interval node
		seng.var(name) : symengine variable
			Symengine variable called 'name'
		"""
		name = str(obj.token.value)
		obj.depth = 0
		intv = Globals.inputVars.get(obj.token.value, None)
		print(intv)
		if intv is not None and intv["INTV"] is None:
			return SymTup((Sym(0.0, Globals.__F__),))
		elif intv is not None and (intv["INTV"][0]==intv["INTV"][1]):
			return SymTup((Sym( intv["INTV"][0], Globals.__T__),))
		else:
			return SymTup((Sym(obj.token.value, Globals.__T__),))


	@staticmethod
	def set_noise(obj, value):
		"""
		# TODO: Check whether the description is correct
		Sets the noise (error) on the free variable.
		Parameters
		----------
		obj : node type
		value : tuple of two numbers (can be integer or float)
			New noise value
		Returns
		-------
		Nothing
		"""
		obj.noise = value

	@staticmethod
	def get_noise(obj):
		"""
		Gets the first value in the noise tuple.
		Parameters
		----------
		obj : node type
		Returns
		-------
		Nothing
		"""
		return abs(obj.noise[0])


	def mutate_to_abstract(self, tvalue, tid):
		"""
		Modify this nodes data to denote this node is being abstracted. Sets the value of the token associated with this
		node to some other value (Strictly a symbolic variable?) and changes the the token type.
		Parameters
		----------
		tvalue : Symengine variable
		tid : string
		"""
		self.token.value = tvalue #SymTup((Sym(tvalue, Globals.__T__), ))
		self.token.type = tid







##-- Var Nodes
class Var(AST):
	"""
	The Var node: Derived from AST representing Input, Output and Expression variables.
	Attributes
	----------
	token : Lexer token object (ID token in the EXPRS section that are not reserved keywords)
	cond : predicate value of this predicated node
	"""
	__slots__ = ['token']
	def __init__(self, token, cond=Globals.__T__):
		super().__init__()
		self.token = token
		self.cond = cond

	def __repr__(self):
		"""
		Returns
		-------
		repr_str : str
			String representation of the class.
		"""
		repr_str = '\nVar{' + '\n\ttoken:' + repr(self.token) + super().__repr__() + '\n}'
		return repr_str

	@staticmethod
	def eval(obj, round_mode="fl64", inv=False):
		"""
		# TODO: Find what inv is.
		Sets rounding and returns the expression at the node.
		Parameters
		----------
		obj : node type
		round_mode : string
			Describes rounding mode at the node.
		inv : ?
		Returns
		-------
		SymTup : Object
			An object of class SymTup containing the value as parsed by the parser and True boolean value for the
			conditional node.
		"""
		nodeList = Globals.global_symbol_table[0]._symTab.get(obj.token.value, None)
		if nodeList is None or len(nodeList)==0:
			return SymTup((Sym(obj.token.value, Globals.__T__),))
		else:
			child_der_tokens = [n.derived_token for n in obj.children]
			obj.derived_token = FLOAT if FLOAT in child_der_tokens else INTEGER
			clist = [n[0].f_expression.__and__(n[1])  for n in obj.nodeList]
			f = lambda x, y: SymConcat(x,y)
			return reduce(f, clist, SymTup((Sym(0.0,Globals.__T__),)))
			#return SymTup( n[0].f_expression.__and__(n[1])  for n in nodeList  )


##-- Creates a lifted node taking a list of (node, conds)
class LiftOp(AST):
	__slots__ = ['token']
	def __init__(self, nodeList, token, cond=Globals.__T__):
		super().__init__()
		self.token = CToken('IF', value="")
#		self.token.type = IF
		self.depth = max([n[0].depth for n in nodeList]) + 1
		self.nodeList = nodeList
		self.children = [n[0] for n in nodeList]
		self.derived_token = FLOAT if FLOAT in [n.derived_token for n in self.children] else INTEGER
		self.cond = cond
		for n in nodeList:
			n[0].parents += (self, )
		#print(nodeList)

	def __repr__(self):
		"""
		Returns
		-------
		repr_str : str
			String representation of the class.
		"""
		repr_str = '\nLiftOp{' + '\n\ttoken:' + repr(self.token) + super().__repr__() + '\n}'
		return repr_str


	@staticmethod
	def eval(obj, inv=False):
		
		clist = [n[0].f_expression.__and__(n[1])  for n in obj.nodeList]
		f = lambda x, y: SymConcat(x,y)
		return reduce(f, clist)

		#return SymTup( n[0].f_expression.__and__(n[1])  for n in obj.nodeList  )

	@staticmethod
	def rec_eval(obj):
		
		obj.depth = max([n[0].depth for n in obj.nodeList]) +1
		clist = [n[0].rec_eval(n[0]).__and__(n[1])  for n in obj.nodeList]
		f = lambda x, y: SymConcat(x,y)
		return reduce(f, clist)


	@staticmethod
	def get_noise(obj):
		return 0.0

##-- Transcendental and special ops
class TransOp(AST):
	"""
	The TransOp node: Derived from AST representing transcendental unary operations.

	Attributes
	----------
	right : node type
		The only child/operand of this node
	token : Lexer token object (SQRT, SIN, ASIN, COS, TAN, EXP)

	Notes
	-----
	These nodes have a single child and always called the right child.
	"""
	__slots__ = ['token']
	def __init__(self, right, token, cond=Globals.__T__):
		super().__init__()
		self.token = token
		self.cond = cond
		self.depth = right.depth+1
		self.children = (right, )
		right.parents += (self, )
		self.derived_token = FLOAT if FLOAT in [n.derived_token for n in self.children] else INTEGER

	def __repr__(self):
		"""
		Returns
		-------
		repr_str : str
			String representation of the class.
		"""
		repr_str = '\nTransOp{' + '\n\ttoken:' + repr(self.token) + super().__repr__() \
				   + '\n\tchildren:' + repr(self.children[0].token.value) \
				   + '\n}'
		return repr_str

	@staticmethod
	def eval(obj, inv=False):
		"""
		Sets rounding, depth and returns simplified symbolic expression without recursively building expression from child.
		Parameters
		----------
		obj : node type
		Returns
		-------
		lexpr : symbolic expression
		"""
		assert isinstance(obj, TransOp)
		lexpr = ops._FOPS[obj.token.type]([obj.children[0].f_expression])
		obj.depth = obj.children[0].depth+1
		obj.rnd = obj.children[0].rnd

		return lexpr

	@staticmethod
	def rec_eval(obj):
		"""
		Sets rounding, recursively builds expression from child and returns simplified expression.
		Parameters
		----------
		obj : node type
		Returns
		-------
		lexpr : symbolic expression
		"""
		lexpr = ops._FOPS[obj.token.type]([obj.children[0].rec_eval(obj.children[0])])
		obj.depth = obj.children[0].depth +1
		obj.rnd = obj.children[0].rnd
		return lexpr

	def get_rounding(self):
		"""
		Returns the rounding value for this node.

		Parameters
		----------
		None

		Returns
		-------
		float
			Product of the rounding factor and ULP for the transcendental operation
		"""
		return self.rnd * ops._ALLOC_ULP[self.token.type]


##-- corresponds to arith binary
class BinOp(AST):
	"""
	The BinOp node: Derived from AST representing binary operations.

	Attributes
	----------
	left : node type
		The left child/operand of this node
	token : Lexer token object (PLUS, MINUS, MUL, DIV)
	right : node type
		The right child/operand of this node

	Notes
	-----
	These nodes have exactly two children.
	"""
	__slots__ = ['token']
	def __init__(self, left, token, right, cond=Globals.__T__):
		super().__init__()
		self.token = token
		self.cond=cond
		self.children = (left, right, )
		self.depth = max(left.depth, right.depth) +1
		left.parents += (self,)
		right.parents += (self, )
		self.derived_token = FLOAT if FLOAT in [n.derived_token for n in self.children] else INTEGER

	def __repr__(self):
		"""
		Returns
		-------
		repr_str : str
			String representation of the class.
		"""
		repr_str = '\nBinOp{' + '\n\ttoken:' + repr(self.token) + super().__repr__() \
				   + '\n\tleft child:' + repr(self.children[0].token.value) \
				   + '\n\tright child:' + repr(self.children[1].token.value) \
				   + '\n}'
		return repr_str

	@staticmethod
	def eval(obj, inv=False):
		"""
		Sets rounding, depth and returns simplified expression without recursively building expression.
		Parameters
		----------
		obj : node type
		Returns
		-------
		lexpr : symbolic expression
		"""
		#print(obj.token.value, [child.f_expression for child in obj.children])
		lexpr = ops._FOPS[obj.token.type]([child.f_expression for child in obj.children])
		obj.depth = max([child.depth for child in obj.children])+1
		obj.rnd = max([min([child.rnd for child in obj.children]), 1.0])

		return lexpr


	@staticmethod
	def rec_eval(obj):
		"""
		Sets rounding, recursively builds expression from children and returns simplified expression..
		Parameters
		----------
		obj : node type
		Returns
		-------
		lexpr : symbolic expression
		"""
		lexpr = ops._FOPS[obj.token.type]([child.rec_eval(child) for child in obj.children])
		obj.depth = max([child.depth for child in obj.children])+1
		obj.rnd = max([min([child.rnd for child in obj.children]), 1.0])

		return lexpr
		

	def get_rounding(self):
		"""
		Returns the rounding value for this node.

		Parameters
		----------
		None

		Returns
		-------
		float
			Product of the rounding factor and ULP for the transcendental operation
		"""
		return self.rnd * ops._ALLOC_ULP[self.token.type]


##-- Binary Literal
class BinLiteral(AST):
	"""
	The BinLiteral node: Derived from AST representing logical binary operations (AND, OR).

	Attributes
	----------
	left : node type
		The left child/operand of this node
	token : Lexer token object (AND, OR)
	right : node type
		The right child/operand of this node

	Notes
	-----
	These nodes have exactly two children.
	"""
	__slots__ = ['token']
	def __init__(self, left, token, right):
		super().__init__()
		self.token = token
		self.children = (left, right, )
		self.depth = max(left.depth, right.depth) +1
		left.parents += (self,)
		right.parents += (self,)

	def __repr__(self):
		"""
		Returns
		-------
		repr_str : str
			String representation of the class.
		"""
		repr_str = '\nBinLiteral{' + '\n\ttoken:' + repr(self.token) + super().__repr__() \
				   + '\n\tleft child:' + repr(self.children[0].token.value) \
				   + '\n\tright child:' + repr(self.children[1].token.value) \
				   + '\n}'
		return repr_str

		#self.f_expression = self.eval(self)

#	@staticmethod
#	def eval(obj):
#		lexpr = ops._BOPS[obj.token.type]([obj.children[0].f_expression, obj.children[1].f_expression])
#		obj.depth = max([child.depth for child in obj.children])+1
#		return lexpr
#
#
#	@staticmethod
#	def rec_eval(obj):
#		lexpr = ops._BOPS[obj.token.type]([obj.children[0].rec_eval(obj.children[0]), obj.children[1].rec_eval(obj.children[1])])
#		obj.depth = max([child.depth for child in obj.children])+1
#		return lexpr


	@staticmethod
	def eval(obj, inv=False):
		"""
		Sets depth and returns simplified expression without recursively building expression.

		Parameters
		----------
		obj : node type

		Returns
		-------
		litexpr : string
			String representation of the condition expression.
		"""
		obj.depth = max([child.depth for child in obj.children])+1
		lstr = obj.children[0].f_expression
		rstr = obj.children[1].f_expression

		# TODO: prints here
		print("LSTR:", lstr)
		print("RSTR:", rstr)

		litexpr = ops._BOPS[obj.token.type if not inv else ops.invert[obj.token.type]]([lstr,rstr])
		#print("LITEXPR:", litexpr)

		return litexpr



	@staticmethod
	def rec_eval(obj):
		"""
		Sets depth and recursively builds expression from children and returns simplified expression.

		Parameters
		----------
		obj : node type

		Returns
		-------
		litexpr : string
			String representation of the condition expression.
		"""
		obj.depth = max([child.depth for child in obj.children])+1
		lstr = obj.children[0].rec_eval(obj.children[0])
		rstr = obj.children[1].rec_eval(obj.children[1])

		litexpr = ops._BOPS[obj.token.type]([lstr,rstr])

		return litexpr

##-- Comparison Operators
class ExprComp(AST):
	"""
	The BinLiteral node: Derived from AST representing comparison operations.

	Attributes
	----------
	token : Lexer token object (EQ, NEQ, LT, LEQ, GT, GEQ)
	condSym : symbolic variable
		Represents the expression for this node and has prefix 'ES' with suffix id assigned by Globals.EID
		eg: ES0, ES1, ...
	"""
	__slots__ = ['token', 'condSym']
	def __init__(self, left, token, right):
		super().__init__()
		self.token=token
		self.children = (left, right,)
		self.depth = max([left.depth, right.depth])+1
		left.parents += (self, )
		right.parents += (self, )
		self.condSym = sym.var("ES"+str(Globals.EID))
		Globals.EID += 1
		Globals.condTable[self.condSym] = self
		self.derived_token = FLOAT if FLOAT in [n[0].derived_token for n in self.nodeList] else INTEGER

		## have the f_expressions evaluted early for conds
		#self.f_expression = self.eval(self)

	def __repr__(self):
		"""
		Returns
		-------
		repr_str : str
			String representation of the class.
		"""
		repr_str = '\nExprComp{' + '\n\ttoken:' + repr(self.token) + super().__repr__() \
				   + '\n\tleft child:' + repr(self.children[0].token.value) \
				   + '\n\tright child:' + repr(self.children[1].token.value) \
				   + '\n}'
		return repr_str

	@staticmethod
	def mod_eval(obj, inv, err0, err1):
		"""
		Sets depth of node and generates a modified (error adjusted) condition expression

		Parameters
		----------
		obj : node type
			Object of this node type
		inv : ?
		err0 : number
			Error to be adjusted in the left side
		err1 : number
			Error to be adjusted in the right side

		Returns
		-------
		str
			Condition expression in string format
		"""
		obj.depth = max([child.depth for child in obj.children])+1
		lstrTup = obj.children[0].f_expression
		rstrTup = obj.children[1].f_expression

		free_syms = lstrTup.__freeSyms__().union(rstrTup.__freeSyms__())
		ERR0 = 0 if obj.children[0].derived_token==INTEGER else err0
		ERR1 = 0 if obj.children[1].derived_token==INTEGER else err1
		#ERR0 = err0
		#ERR1 = err1
		#for child in obj.children:
		#	if child.derived_token==INTEGER:
		#		print("INTEGER DERIVED TOKEN @", obj.token.lineno)
		#		print(child.f_expression, type(child))
		#		print([c.derived_token==FLOAT for c in child.children])

		# Error adjusted condition expression in string form.
		cexpr = tuple(set(ops._MCOPS[obj.token.type if not inv else ops.invert[obj.token.type]]([fl.exprCond[0],\
											 sl.exprCond[0], ERR0, ERR1]) \
											 for fl in lstrTup \
										     for sl in rstrTup))

		if ("(True)" in cexpr):
			return ("<<True>>",set())
		else:
			l1 = list(filter(lambda x: x!="(False)", cexpr))
			#free_syms = reduce(lambda x, y: x.union(y), [el.exprCond[0].free_symbols for el in l1 if (seng.count_ops(el.exprCond[0]) > 0)], set())
			return ("<<False>>",set()) if len(l1)==0 else ("<<{comp_expr}>>".format( comp_expr = ">> | <<".join(l1)), free_syms)


	@staticmethod
	def eval(obj):
		"""
		Sets depth of node and generates a condition expression without recursively building expression from children.

		Parameters
		----------
		obj : node type
			Object of this node type

		Returns
		-------
		str
			Condition expression in string format
		"""
		obj.depth = max([child.depth for child in obj.children])+1
		lstrTup = obj.children[0].f_expression
		rstrTup = obj.children[1].f_expression

		cexpr = tuple(set(ops._COPS[obj.token.type]([fl.exprCond[0],\
											 sl.exprCond[0]]) \
											 for fl in lstrTup \
										     for sl in rstrTup))

		print("Never should be here")
		return "|".join(cexpr)

	@staticmethod
	def rec_eval(obj):
		"""
		Sets depth of node and generates condition expression by recursively building expression from children.
		Parameters
		----------
		obj : node type
			Object of this node type

		Returns
		-------
		str
			Condition expression in string format
		"""
		obj.depth = max([child.depth for child in obj.children])+1
		lstrTup = obj.children[0].rec_eval(obj.children[0])
		rstrTup = obj.children[1].rec_eval(obj.children[1])

		#print(lstrTup, rstrTup)

		cexpr = tuple(set(ops._COPS[obj.token.type]([fl.exprCond[0],\
											 sl.exprCond[0]]) \
											 for fl in lstrTup \
										     for sl in rstrTup))

		return "|".join(cexpr)


	#@staticmethod
	#def eval(obj):
	#	lexpr = ops._COPS[obj.token.type]([obj.children[0].f_expression, obj.children[1].f_expression])
	#	obj.depth = max([child.depth for child in obj.children])+1
	#	return lexpr

	#@staticmethod
	#def rec_eval(obj):
	#	lexpr = ops._COPS[obj.token.type]([obj.children[0].rec_eval(obj.children[0]), obj.children[1].rec_eval(obj.children[1])])
	#	obj.depth = max([child.depth for child in obj.children])+1
	#	return lexpr


