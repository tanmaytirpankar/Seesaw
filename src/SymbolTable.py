
import Globals
import sympy 
import symengine as seng

from PredicatedSymbol import Sym, SymTup, SymConcat
from functools import reduce


class SymbolTable(object):
	"""
	Class associating variables to predicated nodes in the AST. This is one node in a SymbolTable linked list.

	Attributes
	----------
	_symtab : dict
		A dict mapping variables to a tuple of predicated nodes.
	_scope : int
		Scope ID for which this symbol table stands in the program.
	_scopeCond : bool
		Predicate associated with this Symbol Table that can be used for symbol table merging purposes.

	Methods
	-------
	insert(symbol, nodeCondTup)
		Inserts a entry in this Symbol Table with symbol as key and nodeCondTup as value (tuple of predicated nodes)
	lookup(symbol)
		Recursively searches UP (parents) for "symbol" variable entry starting at the current symbol table and returns
		the tuple of predicated nodes associated with it.
	"""
	__slots__ = ['_symTab', '_scope', '_scopeCond']

	def __init__(self, scope=0, cond=Globals.__T__, caller_symTab=None):
		self._scope = scope
		self._symTab = {}
		self._scopeCond = cond
		self._symTab['_caller_'] = caller_symTab

	def insert(self, symbol, nodeCondTup):
		"""
		Inserts an entry in this Symbol Table with symbol as key and nodeCondTup as value (tuple of predicated nodes)
		Returns
		-------
		Nothing
		"""
		self._symTab[symbol] = nodeCondTup

	def lookup(self, symbol):
		"""
		Recursively searches UP (parents) for "symbol" variable entry starting at the current symbol table.
		Returns
		-------
		tuple
			Returns the tuple of predicated nodes associated with this Symbol Table
		"""
		if self._symTab['_caller_'] is None:
			return self._symTab.get(symbol)
			
		return self._symTab.get(symbol, self._symTab['_caller_'].lookup(symbol))
