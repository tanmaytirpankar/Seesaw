import sys

sys.path.insert(1, '../../src')
from lexer import Slex
from PredicatedSymbol import Sym, SymTup, SymConcat
import symengine as seng
from ASTtypes import *
from Globals import *
from SymbolTable import *

class TestNodes:

    def test_num(self):
        lexer = Slex()
        token_generator = lexer.tokenize("12")
        token = token_generator.__next__()
        node = Num(token)
        assert node.depth == 0
        assert node.children == ()
        assert node.parents == ()
        assert node.noise == (0, 0)
        assert node.nodeList == []
        assert node.f_expression == SymTup((Sym(token.value, Globals.__T__),))
        assert node.rnd == 0.0
        assert node.cond == Globals.__T__
        assert node.derived_token == INTEGER
        node.set_expression(SymTup((Sym(14, Globals.__T__),)))
        assert isinstance(node.f_expression, SymTup)

    def test_free_var(self):
        lexer = Slex()
        token_generator = lexer.tokenize("circ")
        token = token_generator.__next__()
        node = FreeVar(token)
        assert node.depth == 0
        assert node.children == ()
        assert node.parents == ()
        assert node.noise == (0, 0)
        assert node.nodeList == []
        assert node.f_expression == None
        assert node.rnd == 1.0
        assert node.token == token
        assert node.cond == Globals.__T__
        assert node.derived_token == FLOAT
        assert node.eval(node) == SymTup((Sym(token.value, Globals.__T__),))
        node.set_noise(node, (1, 0))
        node.noise[0] == 1
        assert node.get_noise(node) == 1
        node.mutate_to_abstract(seng.var("FR0"), ID)
        assert node.token.value == FR0
        assert node.token.type == ID


    def test_var(self):
        lexer = Slex()
        token_generator = lexer.tokenize("circ")
        token = token_generator.__next__()
        node = Var(token, Globals.__F__)
        assert node.depth == 0
        assert node.f_expression == None
        assert node.children == ()
        assert node.parents == ()
        assert node.noise == (0, 0)
        assert node.rnd == 1.0
        assert node.cond == Globals.__F__
        assert node.nodeList == []
        assert node.derived_token == FLOAT
        assert node.token == token
        symTab = SymbolTable()
        Globals.scopeID = Globals.scopeID + 1
        Globals.global_symbol_table[Globals.scopeID] = symTab
        assert node.eval(node) == SymTup((Sym(node.token.value, Globals.__T__),))
        # TODO: Write tests for the else case of eval().

    def test_lift_operation(self):
        lexer = Slex()
        token_generator = lexer.tokenize("circ trian square")
        token = token_generator.__next__()
        symTab = SymbolTable(0)
        symVar1 = FreeVar(token, cond=Globals.__T__)
        token = token_generator.__next__()
        symVar2 = FreeVar(token, cond=Globals.__T__)
        token = token_generator.__next__()
        symTab.insert(token.value, ((symVar1, Globals.__T__), (symVar2, Globals.__T__),))
        lval = symTab.lookup(token.value)
        node = LiftOp(lval, token, Globals.__F__)
        assert node.depth == 1
        assert node.f_expression == None
        assert node.children == [symVar1, symVar2]
        assert node.parents == ()
        assert node.noise == (0, 0)
        assert node.rnd == 1.0
        assert node.cond == Globals.__F__
        assert node.derived_token == FLOAT
        assert isinstance(node.token, CToken)
        assert node.token.value is None

    def test_transcendental_operation(self):
        return

    def test_binary_operation(self):
        return

    def test_binary_literal(self):
        return

    def test_expression_comparison(self):
        return