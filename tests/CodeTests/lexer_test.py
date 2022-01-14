import sys

sys.path.insert(1, '../../src')
from lexer import Slex


class TestTokens:
    def test_INPUTS(self):
        lexer = Slex()
        tok = lexer.tokenize("INPUTS")
        x = tok.__next__()
        assert x.type == 'INPUTS'
        assert x.value == 'INPUTS'

    def test_OUTPUTS(self):
        lexer = Slex()
        tok = lexer.tokenize("OUTPUTS")
        x = tok.__next__()
        assert x.type == 'OUTPUTS'
        assert x.value == 'OUTPUTS'

    def test_EXPRS(self):
        lexer = Slex()
        tok = lexer.tokenize("EXPRS")
        x = tok.__next__()
        assert x.type == 'EXPRS'
        assert x.value == 'EXPRS'

    def test_REQUIRES(self):
        lexer = Slex()
        tok = lexer.tokenize("REQUIRES")
        x = tok.__next__()
        assert x.type == 'REQUIRES'
        assert x.value == 'REQUIRES'

    def test_if(self):
        lexer = Slex()
        tok = lexer.tokenize("if")
        x = tok.__next__()
        print(x)
        assert x.type == 'IF'
        assert x.value == 'if'

    def test_then(self):
        lexer = Slex()
        tok = lexer.tokenize("then")
        x = tok.__next__()
        assert x.type == 'THEN'
        assert x.value == 'then'

    def test_else(self):
        lexer = Slex()
        tok = lexer.tokenize("else")
        x = tok.__next__()
        assert x.type == 'ELSE'
        assert x.value == 'else'

    def test_endif(self):
        lexer = Slex()
        tok = lexer.tokenize("endif")
        x = tok.__next__()
        assert x.type == 'ENDIF'
        assert x.value == 'endif'

    def test_integer(self):
        lexer = Slex()
        tok = lexer.tokenize("12")
        x = tok.__next__()
        assert x.type == 'INTEGER'
        assert x.value == 12
        lexer = Slex()
        tok = lexer.tokenize("-12")
        x = tok.__next__()
        assert x.type == 'INTEGER'
        assert x.value == -12

    def test_float(self):
        lexer = Slex()
        tok = lexer.tokenize("1.0")
        x = tok.__next__()
        assert x.type == 'FLOAT'
        assert x.value == 1.0
        lexer = Slex()
        tok = lexer.tokenize("-1.0e+2")
        x = tok.__next__()
        assert x.type == 'FLOAT'
        assert x.value == -100.0

    def test_sin(self):
        lexer = Slex()
        tok = lexer.tokenize("sin")
        x = tok.__next__()
        assert x.type == 'SIN'
        assert x.value == 'sin'

    def test_asin(self):
        lexer = Slex()
        tok = lexer.tokenize("asin")
        x = tok.__next__()
        assert x.type == 'SIN'
        assert x.value == 'asin'

    def test_cos(self):
        lexer = Slex()
        tok = lexer.tokenize("cos")
        x = tok.__next__()
        assert x.type == 'COS'
        assert x.value == 'cos'

    def test_tan(self):
        lexer = Slex()
        tok = lexer.tokenize("tan")
        x = tok.__next__()
        assert x.type == 'TAN'
        assert x.value == 'tan'

    def test_cot(self):
        lexer = Slex()
        tok = lexer.tokenize("cot")
        x = tok.__next__()
        assert x.type == 'COT'
        assert x.value == 'cot'

    def test_cosh(self):
        lexer = Slex()
        tok = lexer.tokenize("cosh")
        x = tok.__next__()
        assert x.type == 'COSH'
        assert x.value == 'cosh'

    def test_sinh(self):
        lexer = Slex()
        tok = lexer.tokenize("sinh")
        x = tok.__next__()
        assert x.type == 'SINH'
        assert x.value == 'sinh'

    def test_log(self):
        lexer = Slex()
        tok = lexer.tokenize("sqrt")
        x = tok.__next__()
        assert x.type == 'SQRT'
        assert x.value == 'sqrt'

    def test_log(self):
        lexer = Slex()
        tok = lexer.tokenize("log")
        x = tok.__next__()
        assert x.type == 'LOG'
        assert x.value == 'log'

    def test_exp(self):
        lexer = Slex()
        tok = lexer.tokenize("exp")
        x = tok.__next__()
        assert x.type == 'EXP'
        assert x.value == 'exp'

    def test_PLUS(self):
        lexer = Slex()
        tok = lexer.tokenize("+")
        x = tok.__next__()
        assert x.type == 'PLUS'
        assert x.value == '+'

    def test_MINUS(self):
        lexer = Slex()
        tok = lexer.tokenize("-")
        x = tok.__next__()
        assert x.type == 'MINUS'
        assert x.value == '-'

    def test_MUL(self):
        lexer = Slex()
        tok = lexer.tokenize("*")
        x = tok.__next__()
        assert x.type == 'MUL'
        assert x.value == '*'

    def test_DIV(self):
        lexer = Slex()
        tok = lexer.tokenize("/")
        x = tok.__next__()
        assert x.type == 'DIV'
        assert x.value == '/'

    def test_EQ(self):
        lexer = Slex()
        tok = lexer.tokenize("==")
        x = tok.__next__()
        assert x.type == 'EQ'
        assert x.value == '=='

    def test_NEQ(self):
        lexer = Slex()
        tok = lexer.tokenize("!=")
        x = tok.__next__()
        assert x.type == 'NEQ'
        assert x.value == '!='

    def test_LEQ(self):
        lexer = Slex()
        tok = lexer.tokenize("<=")
        x = tok.__next__()
        assert x.type == 'LEQ'
        assert x.value == '<='

    def test_LT(self):
        lexer = Slex()
        tok = lexer.tokenize("<")
        x = tok.__next__()
        assert x.type == 'LT'
        assert x.value == '<'

    def test_GEQ(self):
        lexer = Slex()
        tok = lexer.tokenize(">=")
        x = tok.__next__()
        assert x.type == 'GEQ'
        assert x.value == '>='

    def test_GT(self):
        lexer = Slex()
        tok = lexer.tokenize(">")
        x = tok.__next__()
        assert x.type == 'GT'
        assert x.value == '>'

    def test_AND(self):
        lexer = Slex()
        tok = lexer.tokenize("&&")
        x = tok.__next__()
        assert x.type == 'AND'
        assert x.value == '&&'

    def test_OR(self):
        lexer = Slex()
        tok = lexer.tokenize("||")
        x = tok.__next__()
        assert x.type == 'OR'
        assert x.value == '||'

    def test_NOT(self):
        lexer = Slex()
        tok = lexer.tokenize("!")
        x = tok.__next__()
        assert x.type == 'NOT'
        assert x.value == '!'

    def test_LPAREN(self):
        lexer = Slex()
        tok = lexer.tokenize("(")
        x = tok.__next__()
        assert x.type == 'LPAREN'
        assert x.value == '('

    def test_RPAREN(self):
        lexer = Slex()
        tok = lexer.tokenize(")")
        x = tok.__next__()
        assert x.type == 'RPAREN'
        assert x.value == ')'

    def test_SLPAREN(self):
        lexer = Slex()
        tok = lexer.tokenize("{")
        x = tok.__next__()
        assert x.type == 'SLPAREN'
        assert x.value == '{'

    def test_SRPAREN(self):
        lexer = Slex()
        tok = lexer.tokenize("}")
        x = tok.__next__()
        assert x.type == 'SRPAREN'
        assert x.value == '}'

    def test_ASSIGN(self):
        lexer = Slex()
        tok = lexer.tokenize("=")
        x = tok.__next__()
        assert x.type == 'ASSIGN'
        assert x.value == '='

    def test_COLON(self):
        lexer = Slex()
        tok = lexer.tokenize(":")
        x = tok.__next__()
        assert x.type == 'COLON'
        assert x.value == ':'

    def test_SEMICOLON(self):
        lexer = Slex()
        tok = lexer.tokenize(";")
        x = tok.__next__()
        assert x.type == 'SEMICOLON'
        assert x.value == ';'

    def test_COMMA(self):
        lexer = Slex()
        tok = lexer.tokenize(",")
        x = tok.__next__()
        assert x.type == 'COMMA'
        assert x.value == ','

    def test_FPTYPE(self):
        lexer = Slex()
        tok = lexer.tokenize("rnd16")
        x = tok.__next__()
        assert x.type == 'FPTYPE'
        assert x.value == 'rnd16'
        tok = lexer.tokenize("rnd32")
        x = tok.__next__()
        assert x.type == 'FPTYPE'
        assert x.value == 'rnd32'
        tok = lexer.tokenize("rnd64")
        x = tok.__next__()
        assert x.type == 'FPTYPE'
        assert x.value == 'rnd64'
        tok = lexer.tokenize("fl16")
        x = tok.__next__()
        assert x.type == 'FPTYPE'
        assert x.value == 'fl16'
        tok = lexer.tokenize("fl32")
        x = tok.__next__()
        assert x.type == 'FPTYPE'
        assert x.value == 'fl32'
        tok = lexer.tokenize("fl64")
        x = tok.__next__()
        assert x.type == 'FPTYPE'
        assert x.value == 'fl64'

    def test_INTTYPE(self):
        lexer = Slex()
        tok = lexer.tokenize("int")
        x = tok.__next__()
        assert x.type == 'INTTYPE'
        assert x.value == 'int'
