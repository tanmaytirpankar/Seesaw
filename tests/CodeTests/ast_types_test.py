import sys

sys.path.insert(1, '../../src')
from lexer import Slex
from PredicatedSymbol import Sym, SymTup, SymConcat
import symengine as seng
import ops_def as ops
from ASTtypes import *
from Globals import *
from SymbolTable import *


class TestNodes:
    def test_num(self):
        lexer = Slex()
        token_generator = lexer.tokenize("12")
        token = token_generator.__next__()

        # Testing Node properties
        test_node = Num(token)
        assert test_node.depth == 0
        assert test_node.children == ()
        assert test_node.parents == ()
        assert test_node.noise == (0, 0)
        assert test_node.nodeList == []
        assert test_node.f_expression == SymTup((Sym(token.value, Globals.__T__),))
        assert test_node.rnd == 0.0
        assert test_node.cond == Globals.__T__
        assert test_node.derived_token == INTEGER

        # Testing set_expression function
        test_node.set_expression(SymTup((Sym(14, Globals.__T__),)))
        assert isinstance(test_node.f_expression, SymTup)

    def test_free_var(self):
        lexer = Slex()
        token_generator = lexer.tokenize("circ")
        token = token_generator.__next__()

        # Testing Node properties
        test_node = FreeVar(token)
        assert test_node.depth == 0
        assert test_node.children == ()
        assert test_node.parents == ()
        assert test_node.noise == (0, 0)
        assert test_node.nodeList == []
        assert test_node.f_expression is None
        assert test_node.rnd == 1.0
        assert test_node.token == token
        assert test_node.cond == Globals.__T__
        assert test_node.derived_token == FLOAT
        assert test_node.eval(test_node) == SymTup((Sym(token.value, Globals.__T__),))

        # Testing set_noise and get_noise functions for this node
        test_node.set_noise(test_node, (1, 0))
        assert test_node.get_noise(test_node) == 1

        # Testing mutate_to_abstract function
        test_node.mutate_to_abstract(seng.var("FR0"), ID)
        assert test_node.token.value == FR0
        assert test_node.token.type == ID

    def test_var(self):
        lexer = Slex()
        token_generator = lexer.tokenize("circ")
        token = token_generator.__next__()

        # Testing Node properties
        test_node = Var(token, Globals.__F__)
        assert test_node.depth == 0
        assert test_node.f_expression is None
        assert test_node.children == ()
        assert test_node.parents == ()
        assert test_node.noise == (0, 0)
        assert test_node.rnd == 1.0
        assert test_node.cond == Globals.__F__
        assert test_node.nodeList == []
        assert test_node.derived_token == FLOAT
        assert test_node.token == token

        # Testing eval function for this node
        symbol_table = SymbolTable()
        Globals.scopeID = Globals.scopeID + 1
        Globals.global_symbol_table[Globals.scopeID] = symbol_table
        assert test_node.eval(test_node) == SymTup((Sym(test_node.token.value, Globals.__T__),))
        # TODO: Write tests for the else case of eval().

    def test_lift_operation(self):
        lexer = Slex()
        token_generator = lexer.tokenize("circ triangle square")

        symbol_table = SymbolTable(0)
        token = token_generator.__next__()
        symbolic_variable1 = FreeVar(token, cond=Globals.__T__)

        token = token_generator.__next__()
        symbolic_variable2 = FreeVar(token, cond=Globals.__T__)

        token = token_generator.__next__()
        symbol_table.insert(token.value, ((symbolic_variable1, Globals.__T__), (symbolic_variable2, Globals.__T__),))
        predicated_nodes_list = symbol_table.lookup(token.value)

        # Testing Node properties
        # The token argument is not really needed for this node as the node's token is replaced by an IF token on init
        test_node = LiftOp(predicated_nodes_list, token, Globals.__F__)
        assert test_node.depth == 1
        assert test_node.f_expression is None
        assert test_node.children == [symbolic_variable1, symbolic_variable2]
        assert test_node.parents == ()
        assert symbolic_variable1.parents == (test_node,)
        assert symbolic_variable2.parents == (test_node,)
        assert test_node.noise == (0, 0)
        assert test_node.rnd == 1.0
        assert test_node.cond == Globals.__F__
        assert test_node.derived_token == FLOAT
        assert isinstance(test_node.token, CToken)
        assert test_node.token.value is ""

    def test_transcendental_operation(self):
        lexer = Slex()
        token_generator = lexer.tokenize("sin(90)")
        transcendental_operation_token = token_generator.__next__()
        token_generator.__next__()
        num_node = Num(token_generator.__next__())

        # Testing Node properties
        test_node = TransOp(num_node, transcendental_operation_token)
        assert test_node.depth == 1
        assert test_node.f_expression is None
        assert test_node.children == (num_node,)
        assert test_node.parents == ()
        assert num_node.parents == (test_node,)
        assert test_node.noise == (0, 0)
        assert test_node.rnd == 1.0
        assert test_node.cond == Globals.__T__
        assert test_node.nodeList == []
        assert test_node.derived_token == INTEGER
        assert test_node.token == transcendental_operation_token

        # Testing eval function for this node
        assert test_node.eval(test_node) == ops._FOPS[SIN]([test_node.children[0].f_expression])

        # Testing rec_eval function for this node
        assert test_node.rec_eval(test_node) == ops._FOPS[SIN]([test_node.children[0].f_expression])

        # Testing get_rounding function for this node
        assert test_node.get_rounding() == 0.0

    def test_binary_operation(self):
        lexer = Slex()
        token_generator = lexer.tokenize("2.0+3")
        left_num_node = Num(token_generator.__next__())
        binary_operation_token = token_generator.__next__()
        right_num_node = Num(token_generator.__next__())

        # Testing Node properties
        test_node = BinOp(left_num_node, binary_operation_token, right_num_node)
        assert test_node.depth == 1
        assert test_node.f_expression is None
        assert test_node.children == (left_num_node, right_num_node,)
        assert test_node.parents == ()
        assert left_num_node.parents == (test_node,)
        assert right_num_node.parents == (test_node,)
        assert test_node.noise == (0, 0)
        assert test_node.rnd == 1.0
        assert test_node.cond == Globals.__T__
        assert test_node.nodeList == []
        assert test_node.derived_token == FLOAT
        assert test_node.token == binary_operation_token

        # Testing eval function for this node
        assert test_node.eval(test_node) == ops._FOPS[PLUS]([test_node.children[0].f_expression,
                                                             test_node.children[1].f_expression])

        # Testing rec_eval function for this node
        assert test_node.rec_eval(test_node) == ops._FOPS[PLUS]([test_node.children[0].f_expression,
                                                                 test_node.children[1].f_expression])

        # Testing get_rounding function for this node
        assert test_node.get_rounding() == 1.0

    def test_binary_literal(self):
        lexer = Slex()
        token_generator = lexer.tokenize("2>3 && 3<5")

        # Left side of &&
        left_expr_left_operand = Num(token_generator.__next__())
        left_expr_token = token_generator.__next__()
        left_expr_right_operand = Num(token_generator.__next__())
        left_expr_comp_node = ExprComp(left_expr_left_operand, left_expr_token, left_expr_right_operand)
        left_expr_comp_node.f_expression = left_expr_comp_node.mod_eval(left_expr_comp_node, False, 0.0, 0.0)[0]

        logical_operation_token = token_generator.__next__()

        # Right side of &&
        right_expr_left_operand = Num(token_generator.__next__())
        right_expr_token = token_generator.__next__()
        right_expr_right_operand = Num(token_generator.__next__())
        right_expr_comp_node = ExprComp(right_expr_left_operand, right_expr_token, right_expr_right_operand)
        right_expr_comp_node.f_expression = right_expr_comp_node.mod_eval(right_expr_comp_node, False, 0.0, 0.0)[0]

        # Testing Node properties
        test_node = BinLiteral(left_expr_comp_node, logical_operation_token, right_expr_comp_node)
        assert test_node.depth == 2
        assert test_node.f_expression is None
        assert test_node.children == (left_expr_comp_node, right_expr_comp_node,)
        assert test_node.parents == ()
        assert left_expr_comp_node.parents == (test_node,)
        assert right_expr_comp_node.parents == (test_node,)
        assert test_node.noise == (0, 0)
        assert test_node.rnd == 1.0
        assert test_node.cond == Globals.__T__
        assert test_node.nodeList == []
        assert test_node.derived_token == FLOAT
        assert test_node.token == logical_operation_token

        # Testing eval function for this node
        assert test_node.eval(test_node) == ops._BOPS[logical_operation_token.type if not False else
        ops.invert[logical_operation_token.type]]([left_expr_comp_node.f_expression,
                                                   right_expr_comp_node.f_expression])

        # Testing rec_eval function for this node
        assert test_node.rec_eval(test_node) == ops._BOPS[logical_operation_token.type if not False else
        ops.invert[logical_operation_token.type]]([left_expr_comp_node.f_expression,
                                                   right_expr_comp_node.f_expression])

    def test_expression_comparison(self):
        lexer = Slex()
        token_generator = lexer.tokenize("2.0>3")
        left_num_node = Num(token_generator.__next__())
        expression_comparison_token = token_generator.__next__()
        right_num_node = Num(token_generator.__next__())

        # Testing Node properties
        test_node = ExprComp(left_num_node, expression_comparison_token, right_num_node)
        assert test_node.depth == 1
        assert test_node.f_expression is None
        assert test_node.children == (left_num_node, right_num_node,)
        assert test_node.parents == ()
        assert left_num_node.parents == (test_node,)
        assert right_num_node.parents == (test_node,)
        assert test_node.noise == (0, 0)
        assert test_node.rnd == 1.0
        assert test_node.cond == Globals.__T__
        assert test_node.nodeList == []
        assert test_node.derived_token == INTEGER
        assert test_node.token == expression_comparison_token
        assert test_node.condSym == seng.var("ES2")
        assert Globals.condTable[test_node.condSym] == test_node

        # Testing eval function for this node
        assert test_node.mod_eval(test_node, False, 0.0, 0.0) == ('<<False>>', set())

        # Testing eval function for this node
        assert test_node.eval(test_node) == "(False)"

        # Testing rec_eval function for this node
        assert test_node.rec_eval(test_node) == "(False)"
