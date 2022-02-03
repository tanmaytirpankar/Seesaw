import sys

sys.path.insert(1, '../../src')
from gtokens import *
from Globals import *
from PredicatedSymbol import *
import ops_def as ops
import symengine
import math


class TestOps:
    def test_atomic_plus(self):
        x = symengine.symbols('a')
        operand1 = SymTup((Sym(x, Globals.__T__),))
        operand2 = SymTup((Sym(1.0, Globals.__T__),))

        atomic_conditon_number = ops._atomic_condition_ops[PLUS][0]([operand1, operand2])
        assert atomic_conditon_number == SymTup((Sym((x/(x+1.0)).__abs__(), Globals.__T__),))

        atomic_conditon_number = ops._atomic_condition_ops[PLUS][1]([operand1, operand2])
        assert atomic_conditon_number == SymTup((Sym((1.0/(x+1.0)).__abs__(), Globals.__T__),))

    def test_atomic_minus(self):
        x = symengine.symbols('a')
        operand1 = SymTup((Sym(x, Globals.__T__),))
        operand2 = SymTup((Sym(1.0, Globals.__T__),))

        atomic_conditon_number = ops._atomic_condition_ops[MINUS][0]([operand1, operand2])
        assert atomic_conditon_number == SymTup((Sym((x/(x-1.0)).__abs__(), Globals.__T__),))

        atomic_conditon_number = ops._atomic_condition_ops[MINUS][1]([operand1, operand2])
        assert atomic_conditon_number == SymTup((Sym((1.0/(x-1.0)).__abs__(), Globals.__T__),))

    def test_atomic_mul(self):
        operand1 = SymTup((Sym(1.0, Globals.__T__),))
        operand2 = SymTup((Sym(1.0, Globals.__T__),))

        atomic_conditon_number = ops._atomic_condition_ops[MUL][0]([operand1, operand2])
        assert atomic_conditon_number == SymTup((Sym(1.0, Globals.__T__),))

        atomic_conditon_number = ops._atomic_condition_ops[MUL][1]([operand1, operand2])
        assert atomic_conditon_number == SymTup((Sym(1.0, Globals.__T__),))

    def test_atomic_div(self):
        operand1 = SymTup((Sym(1.0, Globals.__T__),))
        operand2 = SymTup((Sym(1.0, Globals.__T__),))

        atomic_conditon_number = ops._atomic_condition_ops[DIV][0]([operand1, operand2])
        assert atomic_conditon_number == SymTup((Sym(1.0, Globals.__T__),))

        atomic_conditon_number = ops._atomic_condition_ops[DIV][1]([operand1, operand2])
        assert atomic_conditon_number == SymTup((Sym(1.0, Globals.__T__),))

    def test_atomic_sqrt(self):
        operand1 = SymTup((Sym(1.0, Globals.__T__),))

        atomic_conditon_number = ops._atomic_condition_ops[SQRT][0]([operand1])
        assert atomic_conditon_number == SymTup((Sym(0.5, Globals.__T__),))

    def test_atomic_exp(self):
        x = symengine.symbols('a')
        operand1 = SymTup((Sym(x, Globals.__T__),))

        atomic_conditon_number = ops._atomic_condition_ops[EXP][0]([operand1])
        assert atomic_conditon_number == SymTup((Sym(x.__abs__(), Globals.__T__),))

    def test_atomic_sin(self):
        operand1 = SymTup((Sym(0, Globals.__T__),))

        atomic_conditon_number = ops._atomic_condition_ops[SIN][0]([operand1])
        assert atomic_conditon_number == SymTup((Sym(0.0, Globals.__T__),))

    def test_atomic_asin(self):
        operand1 = SymTup((Sym(0.0, Globals.__T__),))

        atomic_conditon_number = ops._atomic_condition_ops[ASIN][0]([operand1])
        assert atomic_conditon_number == SymTup((Sym(0.0, Globals.__T__),))

    def test_atomic_cos(self):
        operand1 = SymTup((Sym(0.0, Globals.__T__),))

        atomic_conditon_number = ops._atomic_condition_ops[COS][0]([operand1])
        assert atomic_conditon_number == SymTup((Sym(0.0, Globals.__T__),))

    def test_atomic_tan(self):
        operand1 = SymTup((Sym(0.0, Globals.__T__),))

        atomic_conditon_number = ops._atomic_condition_ops[TAN][0]([operand1])
        assert atomic_conditon_number == SymTup((Sym(0.0, Globals.__T__),))