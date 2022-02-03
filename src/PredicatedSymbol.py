import symengine as seng
import sympy 
import Globals
import numbers

import time

from functools import reduce
from collections.abc import Iterable


opLimit = 0

# TODO: Is this class needed? No usages found
class PredSymbol(object):
	__slots__ = ['exprCond']

	def __init__(self, expr, cond=Globals.__T__):
		self.exprCond = (expr, cond)

	def __add__(self, obj):
		symexpr = (self.exprCond[0] + obj.exprCond[0])
		return PredSymbol( symexpr.expand() if seng.count_ops(symexpr) > opLimit else symexpr , \
				 (self.exprCond[1] & obj.exprCond[1]).simplify())

	def __sub__(self, obj):
		symexpr = (self.exprCond[0] - obj.exprCond[0])
		return PredSymbol( symexpr.expand() if seng.count_ops(symexpr) > opLimit else symexpr , \
				 (self.exprCond[1] & obj.exprCond[1]).simplify())

	def __mul__(self, obj):
		symexpr = (self.exprCond[0] * obj.exprCond[0])
		return PredSymbol( symexpr.expand() if seng.count_ops(symexpr) > opLimit else symexpr , \
				 (self.exprCond[1] & obj.exprCond[1]).simplify())

	def __truediv__(self, obj):
		symexpr = (self.exprCond[0] / obj.exprCond[0])
		return PredSymbol( symexpr.expand() if seng.count_ops(symexpr) > opLimit else symexpr , \
				 (self.exprCond[1] & obj.exprCond[1]).simplify())

	def __floordiv__(self, obj):
		symexpr = (self.exprCond[0] // obj.exprCond[0])
		return PredSymbol( symexpr.expand() if seng.count_ops(symexpr) > opLimit else symexpr , \
				 (self.exprCond[1] &  obj.exprCond[1]).simplify())

	def __mod__(self, obj):
		return PredSymbol( (self.exprCond[0] % obj.exprCond[0]).expand() , \
				 (self.exprCond[1] & obj.exprCond[1]).simplify())

	def __neg__(self):
		return PredSymbol( (-self.exprCond[0],
							self.exprCond[1]))

	def __cos__(self):
		return PredSymbol( (seng.cos(self.exprCond[0]),	\
							self.exprCond[1]) )

	def __sin__(self):
		return PredSymbol( (seng.sin(self.exprCond[0]),	\
							self.exprCond[1]) )

	def __exp__(self):
		return PredSymbol( (seng.exp(self.exprCond[0]),	\
							self.exprCond[1]) )

	def __and__(self, condSym):
		return PredSymbol( (self.exprCond[0], (exprCond[1] & condSym).simplify()) )

	def __str__(self):
		return '(Expr: {expr} , Cond: {cond})'.format( \
												expr = self.exprCond[0], \
												cond = self.exprCond[1] \
												)

	def __repr__(self):
		return self.__str__()

# TODO: Is this class needed? No usages found
class PSList(list):
	def __init__(self, *args, **kwargs):
		super(PSList, self).__init__(args[0])

	def __add__(self, other):
		return PSList([self[i]+other[j] for i in range(len(self)) for j in range(len(other))])

	def __sub__(self, other):
		return PSList([self[i]-other[j] for i in range(len(self)) for j in range(len(other))])

	def __mul__(self, other):
		return PSList([self[i]*other[j] for i in range(len(self)) for j in range(len(other))])

	def __truediv__(self, other):
		return PSList([self[i]/other[j] for i in range(len(self)) for j in range(len(other))])

	def __floordiv__(self, other):
		return PSList([self[i]//other[j] for i in range(len(self)) for j in range(len(other))])

	def __mod__(self, other):
		return PSList([self[i]%other[j] for i in range(len(self)) for j in range(len(other))])

	def __neg__(self):
		return PSList([-self[i] for i in range(len(self))])

	def __cos__(self):
		return PSList([self[i].__cos__() for i in range(len(self))])

	def __sin__(self):
		return PSList([self[i].__sin__() for i in range(len(self))])

	def __exp__(self):
		return PSList([self[i].__exp__() for i in range(len(self))])

	def __and__(self, condSym):
		return PSList([self[i].__and__(condSym) for i in range(len(self))])

	def __concat__(self, other):
		return PSList(list(self) + list(other))

	def __str__(self):
		return '[ {exp_cond_list} ]'.format( \
							exp_cond_list = ','.join([self[i].__str__() for i in range(len(self))]) \
							)

	def __repr__(self):
		return self.__str__()


class Sym(object):
	"""
	Represents Predicated Expressions. Contains the symbolic expressions and the associated predicates.

	Attributes
	----------
	exprCond : tuple
		A tuple of expression and condition.
	"""
	def __init__(self, expr='', cond=Globals.__T__):
		self.exprCond = (expr, cond)

	#def __add__(self, obj):
	#	#symexpr = self.exprCond[0] + obj.exprCond[0]
	#	return Sym( self.exprCond[0]+obj, self.exprCond[1]) if isinstance(obj, numbers.Number) else \
	#	Sym( seng.expand(self.exprCond[0] + obj.exprCond[0]) if seng.count_ops(self.exprCond[0] + obj.exprCond[0]) < 0 else self.exprCond[0] + obj.exprCond[0]	,\
	#			(self.exprCond[1] & obj.exprCond[1]).simplify() )

	def __add__(self, obj):
		if not isinstance(obj, Sym) :
			return Sym( self.exprCond[0]+obj, self.exprCond[1])
		elif not Globals.simplify:
			expr1 = (self.exprCond[0] + obj.exprCond[0])
			cond = (self.exprCond[1] & obj.exprCond[1]).simplify() 
			return Sym(expr1, cond)			
		else:
			expr1 = (self.exprCond[0] + obj.exprCond[0])
			expr2 = seng.expand(expr1)
			cond = (self.exprCond[1] & obj.exprCond[1]).simplify() 
			op1 = seng.count_ops(expr1)
			op2 = seng.count_ops(expr2)
			if (op2 - op1 > 1000):
				Globals.simplify = False
			return Sym(expr1, cond) if op1 < op2 else Sym(expr2, cond) ;

	def __sub__(self, obj):
		#if isinstance(obj, numbers.Number) :
		if not isinstance(obj, Sym) :
			return Sym( self.exprCond[0]-obj, self.exprCond[1])
		elif not Globals.simplify:
			expr1 = (self.exprCond[0] - obj.exprCond[0])
			cond = (self.exprCond[1] & obj.exprCond[1]).simplify() 
			return Sym(expr1, cond)			
		else:
			expr1 = (self.exprCond[0] - obj.exprCond[0])
			cond = (self.exprCond[1] & obj.exprCond[1]).simplify() 
			expr2 = seng.expand(expr1)
			op1 = seng.count_ops(expr1)
			op2 = seng.count_ops(expr2)
			if (op2 - op1 > 1000):
				Globals.simplify = False
			return Sym(expr1, cond) if op1 < op2 else Sym(expr2, cond) ;

	def __mul__(self, obj):
		return Sym( self.exprCond[0]*obj, self.exprCond[1]) if not isinstance(obj, Sym) else \
		Sym( self.exprCond[0] * obj.exprCond[0]	,\
				(self.exprCond[1] & obj.exprCond[1]).simplify() )

	def __truediv__(self, obj):
		symexpr = self.exprCond[0] / obj.exprCond[0]
		return Sym( seng.expand(symexpr) if seng.count_ops(symexpr) < opLimit else symexpr	,\
				(self.exprCond[1] & obj.exprCond[1]).simplify() )

	def __floordiv__(self, obj):
		symexpr = self.exprCond[0] // obj.exprCond[0]
		return Sym( seng.expand(symexpr) if seng.count_ops(symexpr) < opLimit else symexpr	,\
				(self.exprCond[1] & obj.exprCond[1]).simplify() )

	def __mod__(self, obj):
		symexpr = self.exprCond[0] % obj.exprCond[0]
		return Sym( seng.expand(symexpr) if seng.count_ops(symexpr) < opLimit else symexpr	,\
				(self.exprCond[1] & obj.exprCond[1]).simplify() )

	def __neg__(self):
		return Sym( -self.exprCond[0],
					self.exprCond[1])

	def __tan__(self):
		return Sym( seng.tan(self.exprCond[0])	,\
				self.exprCond[1]  )

	def __cos__(self):
		return Sym( seng.cos(self.exprCond[0])	,\
				self.exprCond[1]  )

	def __sin__(self):
		return Sym( seng.sin(self.exprCond[0])	,\
				self.exprCond[1] )

	def __asin__(self):
		return Sym( seng.asin(self.exprCond[0])	,\
				self.exprCond[1] )

	def __exp__(self):
		return Sym( seng.exp(self.exprCond[0])	,\
				self.exprCond[1] )

	def __and__(self, condSym):
		return Sym( self.exprCond[0], (self.exprCond[1] & condSym) )

	def __abs__(self):
		return Sym( self.exprCond[0].__abs__(), self.exprCond[1] )

	def __sqrt__(self):
		return Sym( seng.sqrt(self.exprCond[0]), self.exprCond[1] )

	def __pow__(self, n):
		return Sym( (self.exprCond[0])**n, self.exprCond[1] )

	def __eq__(self, other):
		return True if(self.exprCond[1]==other.exprCond[1] and \
			           self.exprCond[0]==other.exprCond[0] \
			            ) \
			         else False

	def __countops__(self):
		return seng.count_ops(self.exprCond[0])

	def __Bcountops__(self):
		return sympy.count_ops(self.exprCond[1])


	def __freeSyms__(self):
		try:
			return self.exprCond[0].free_symbols
		except:
			return set()

	def __lt__(self, other):
		assert(self.exprCond[1]==other.exprCond[1])
		return (self.exprCond[0] < other.exprCond[0]) 
		

	def __leq__(self, other):
		assert(self.exprCond[1]==other.exprCond[1])
		return (self.exprCond[0] <= other.exprCond[0])

	def __hash__(self):
		return hash(tuple(self.exprCond))



	def __str__(self):
		return '(Expr: {expr} , Cond : {cond})'.format( \
												expr = self.exprCond[0], \
												cond = self.exprCond[1] \
											)

	def __repr__(self):
		return self.__str__()


def condmerge(tupleList):
		ld = {}
		for els in tupleList:
			(expr,cond) = els.exprCond
			ld[cond] = ld.get(cond, SymTup((Sym(0.0,Globals.__T__),))) + SymTup((els,))
		
		#for k,v in ld.items():
		#	print("//-- Cond =", k)
		#	print(v,"\n")
		#print("Size,", len(tupleList))

		return reduce(lambda x,y : x.__concat__(y,trim=True), [v for k,v in ld.items()], SymTup((Sym(0.0,Globals.__T__),)))

# A Class representing Predicated Symbols. Contains the expression and the associated predicated with operations defined
# for Predicated symbols.
class SymTup(tuple):
	"""
	Represents the value of a node in the Conditional Computation Graph. The value is a tuple of Predicated expressions
	(Sym).

	Attributes
	----------
	Tuple
		A tuple of predicated expressions.
	"""
	def __new__(self, tup):
		return tuple.__new__(SymTup, tup)

	def __add__(self, obj):
		s =  SymTup((fl+obj for fl in self)) if isinstance(obj, numbers.Number) else \
		  SymTup(sel for sel in (fl+sl for fl in tuple(set(el for el in self if  not (el.exprCond[1]==Globals.__F__  or el.exprCond[0]==seng.nan ))) \
		                         for sl in tuple(set(el for el in obj if  not ( el.exprCond[1]==Globals.__F__  or el.exprCond[0]==seng.nan )))) \
								 if  not ( sel.exprCond[1]==Globals.__F__  or sel.exprCond[0]==seng.nan))

		return (s) if(len(s)!=0) else SymTup((Sym(0.0, Globals.__T__),))
		  #SymTup((fl+sl for fl in self for sl in obj))

	def __sub__(self, obj):
		s =  SymTup(sel for sel in (fl-sl for fl in tuple(set(el for el in self if  not ( el.exprCond[1]==Globals.__F__  or el.exprCond[0]==seng.nan))) \
		                      for sl in tuple(set(el for el in obj if  not (el.exprCond[1]==Globals.__F__  or el.exprCond[0]==seng.nan )))) \
							  if  not ( sel.exprCond[1]==Globals.__F__  or sel.exprCond[0]==seng.nan))
		#return s
		return (s) if(len(s)!=0) else SymTup((Sym(0.0, Globals.__T__),))

	def __mul__(self, obj):
		st1 = time.time()
		#t1 = tuple(set(el for el in self if  not ( el.exprCond[1]==Globals.__F__  or el.exprCond[0]==seng.nan )))
		t1 = tuple(set(filter(lambda x: not (x.exprCond[1]==Globals.__F__  or x.exprCond[0]==seng.nan), self)))
		#if isinstance(obj, numbers.Number):
		if not isinstance(obj, Iterable):
			s = SymTup((fl*obj for fl in t1))
		else:
			#print(type(obj))
			t2 = tuple(set(filter(lambda x: not (x.exprCond[1]==Globals.__F__  or x.exprCond[0]==seng.nan), obj)))
			s = SymTup(sel for sel in (fl*sl for fl in t1 for sl in t2) if not ( sel.exprCond[1]==Globals.__F__ or sel.exprCond[0]==seng.nan))
		et1 = time.time()
		#return condmerge(s)
		return condmerge(s) if(len(s)!=0) else SymTup((Sym(0.0, Globals.__T__),))

	#def __truediv__(self, obj):
	#	return  SymTup((fl/sl for fl in self for sl in obj))

	def __truediv__(self, obj):
		t1 = tuple(set(el for el in self if  not ( el.exprCond[1]==Globals.__F__  or el.exprCond[0]==seng.nan )))
		#if isinstance(obj, numbers.Number):
		if not isinstance(obj, Iterable):
			s = SymTup((fl/obj for fl in t1))
		else:
			t2 = tuple(set(el for el in obj if  not ( el.exprCond[1]==Globals.__F__ or el.exprCond[0]==0.0 or el.exprCond[0]==0 or el.exprCond[0]==seng.nan )))
			s = SymTup(sel for sel in (fl/sl for fl in t1 for sl in t2) if not ( sel.exprCond[1]==Globals.__F__ or sel.exprCond[0]==seng.nan))
		#return s
		return (s) if(len(s)!=0) else SymTup((Sym(0.0, Globals.__T__),))

	def __floordiv__(self, obj):
		return  SymTup((fl//sl for fl in self for sl in obj))

	def __mod__(self, obj):
		return  SymTup((fl%sl for fl in self for sl in obj))

	def __neg__(self):
		t1 = tuple(set(el for el in self if
					   not (el.exprCond[1] == Globals.__F__ or el.exprCond[1] == False or el.exprCond[0] == seng.nan)))
		return SymTup((-fl for fl in t1))

	def __tan__(self):
		t1 = tuple(set(el for el in self if  not (el.exprCond[1]==Globals.__F__ or  el.exprCond[1]==False or el.exprCond[0]==seng.nan)))
		return  SymTup((fl.__tan__() for fl in t1))

	def __cos__(self):
		t1 = tuple(set(el for el in self if  not (el.exprCond[1]==Globals.__F__ or  el.exprCond[1]==False or el.exprCond[0]==seng.nan )))
		return  SymTup((fl.__cos__() for fl in t1))

	def __sin__(self):
		t1 = tuple(set(el for el in self if  not (el.exprCond[1]==Globals.__F__ or  el.exprCond[1]==False or el.exprCond[0]==seng.nan )))
		return  SymTup((fl.__sin__() for fl in t1))

	def __asin__(self):
		return  SymTup((fl.__asin__() for fl in self))

	def __exp__(self):
		return  SymTup((fl.__exp__() for fl in self))

	def __and__(self, condSym):
		t1 = tuple(set(el for el in self if  not ( el.exprCond[1]==Globals.__F__ or  el.exprCond[1]==False or el.exprCond[0]==seng.nan )))
		return SymTup((fl.__and__(condSym) for fl in t1))

	def __abs__(self):
		return SymTup((fl.__abs__() for fl in self))

	def __sqrt__(self):
		t1 = tuple(set(el for el in self if  not ( el.exprCond[1]==Globals.__F__ or  el.exprCond[1]==False or el.exprCond[0]==seng.nan )))
		return SymTup((fl.__sqrt__() for fl in t1))

	def __pow__(self, n):
		t1 = tuple(set(el for el in self if  not ( el.exprCond[1]==Globals.__F__ or  el.exprCond[1]==False or el.exprCond[0]==seng.nan )))
		return SymTup((fl.__pow__(n) for fl in t1))

	def __countops__(self):
		return max([fl.__countops__() for fl in self]+[0])

	def __Bcountops__(self):
		return max([fl.__Bcountops__() for fl in self]+[0])

	def __concat__(self, other, trim=False):
		if trim:
			st1 = time.time()
			#t1 = tuple(set(el for el in self if  not (el.exprCond[0]==0 or el.exprCond[0]==seng.nan or el.exprCond[1]==Globals.__F__ or  el.exprCond[1]==False)))
			#t2 = tuple(set(el for el in other if  not (el.exprCond[0]==0 or el.exprCond[0]==seng.nan or el.exprCond[1]==Globals.__F__ or  el.exprCond[1]==False)))
			t1 = tuple(set(filter(lambda x: not (x.exprCond[0]==0 or x.exprCond[0]==seng.nan or x.exprCond[1]==Globals.__F__ or  x.exprCond[1]==False), self)))
			t2 = tuple(set(filter(lambda x: not (x.exprCond[0]==0 or x.exprCond[0]==seng.nan or x.exprCond[1]==Globals.__F__ or  x.exprCond[1]==False), other)))
			#s = SymTup(tuple(set(tuple(t1) + tuple(t2))))
			s = (SymTup(sel for sel in tuple(set(tuple(t1) + tuple(t2))) if  not (sel.exprCond[0]==0 or sel.exprCond[0]==seng.nan or sel.exprCond[1]==Globals.__F__ or  sel.exprCond[1]==False)))
			#return (s)
			return (s) if(len(s)!=0) else SymTup((Sym(0.0, Globals.__T__),))
		else:
			return (SymTup(tuple(self) + tuple(other)))

	def __freeSyms__(self):
		return reduce(lambda x,y: x.union(y), [sl.__freeSyms__() for sl in self], set())

	def __str__(self):
		return 'Tup( {expr_cond_list} )'.format( \
				expr_cond_list = ','.join([self[i].__str__() for i in range(len(self))]) \
				)

	def __repr__(self):
		return self.__str__()


def	SymConcat(t1, t2):
	## atleast either t1 or t2 must be non-empry
	return t1 if len(t2)==0 else t2 if len(t1)==0 else SymTup(tuple(t1) + tuple(t2))


def lambda_add():
	return lambda x : x[0] + x[1]


if __name__ == "__main__":
	x = sym.symbols('x')
	y = sym.symbols('y')
	z = sym.symbols('z')

	a1 = PredSymbol((x+y)*(x-y), ~x & y)
	a2 = PredSymbol((x+y-z*x), (~y | ~x))
	a3 = PredSymbol((3.5), (~z | ~x))

	l1 = PSList([a1,a2])
	l2 = PSList([a3])

	res = lambda_add()
	print("a1 :", a1)
	#print("overload-sym:", res([a1,a2]))
	print("overload-sym:", a1/a2)
	#print(type(l1), type(l2))
	print("overload symList:", ((l1+l2)/(l1*l2)))
	#print("carry forward symList:", (l1+l2)+(l1))
	print("type:", type(l1+l2))
	
