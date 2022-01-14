
from gtokens import *
import symengine as seng
from sly import Lexer


class Slex(Lexer):

	# Tokens declared in the same order as defined in gtokens.py
	tokens = {
		INPUTS, OUTPUTS, EXPRS, REQUIRES,
		IF, THEN, ELSE, ENDIF,
		INTEGER, FLOAT,
		SIN, ASIN, COS, TAN, COT, COSH, SINH,
		SQRT, LOG, EXP, IDEN,
		PLUS, MINUS, MUL, DIV,
		EQ, NEQ, LEQ, LT, GEQ, GT,
		AND, OR, NOT,
		ASSIGN, COLON, SEMICOLON, COMMA, ID,
		LPAREN, RPAREN, SLPAREN, SRPAREN,
		FPTYPE, INTTYPE,
	}

	# Ignore spaces, tabs and anything followed by #
	ignore = ' \t'
	ignore_comment = r'\#.*'

	# regular expressions
	PLUS		=	r'\+'
	MUL			=	r'\*'
	DIV			=	r'\/'
	EQ			=	r'\=='
	NEQ			=	r'\!='
	LEQ 		=	r'\<='
	LT 			=	r'\<'
	GEQ 		=	r'\>='
	GT 			=	r'\>'
	AND 		=	r'\&&'
	OR 			=	r'\|\|'
	NOT 		=	r'\!'
	LPAREN 		=	r'\('
	RPAREN 		=	r'\)'
	SLPAREN 	=	r'\{'
	SRPAREN 	=	r'\}'
	ASSIGN		=	r'\='
	COLON		=	r'\:'
	SEMICOLON	=	r'\;'
	COMMA		=	r'\,'



	# Regular expression for ID
	ID				=	r'[a-zA-Z][a-zA-Z0-9_]*'

	# Remapping some reserved words recognized as ID to the correct tokens.
	ID['INPUTS']	=	INPUTS
	ID['OUTPUTS']	=	OUTPUTS
	ID['EXPRS']		=	EXPRS
	ID['REQUIRES']	=	REQUIRES
	ID['if'] 		= 	IF
	ID['then'] 		= 	THEN
	ID['else'] 		= 	ELSE
	ID['endif'] 	= 	ENDIF
	ID['sin']		= 	SIN
	ID['asin']		= 	SIN
	ID['cos']		= 	COS
	ID['tan']		= 	TAN
	ID['cot']		= 	COT
	ID['cosh']		= 	COSH
	ID['sinh']		= 	SINH
	ID['sqrt'] 		= 	SQRT
	ID['log'] 		= 	LOG
	ID['exp'] 		= 	EXP

	# ['rnd16', 'rnd32', 'rnd64', 'fl16', 'fl32', 'fl64'] = FPTYPE
	ID['rnd16'] 	= 	FPTYPE
	ID['rnd32'] 	= 	FPTYPE
	ID['rnd64'] 	= 	FPTYPE
	ID['fl16']  	= 	FPTYPE
	ID['fl32']  	= 	FPTYPE
	ID['fl64']  	= 	FPTYPE
	ID['int']		= 	INTTYPE

	current_token = None
	tok = None

	# Converting all values recognized as IDs to symbolic values.
	def ID(self, t):
		if t.type not in  (INPUTS, OUTPUTS, EXPRS, REQUIRES):
			t.value = seng.var(t.value)
		return t

	# @_() Decorator used to specify regular expression for FLOAT. Converting value to float.
	@_(r'[\-]?\d+\.\d+([eE][-+]?\d+)?')
	def FLOAT(self, t):
		t.value = float(t.value)
		return t

	# @_() Decorator used to specify regular expression for INTEGER. Converting value to int.
	@_(r'[\-]?\d+')
	def INTEGER(self, t):
		t.value = int(t.value)
		t.type = INTEGER
		return t

	# Placed after FLOAT and INTEGER so negative values are not ignored.
	MINUS	=	r'\-'

	# Reg ex for one or more new lines. Increment line number.
	@_(r'\n+')
	def ignore_newline(self, t):
		self.lineno += t.value.count('\n')

	# Message to be printed when character not recognized. Overrides base method.
	def error(self, t):
		print('Line %d: Bad character %r' % (self.lineno, t.value[0]))

	def create_token_generator(self, text):
		self.tok = self.tokenize(text)

	def get_current_token(self):
		return self.current_token

	def get_next_token(self):
		try:
			return self.tok.__next__()
		except StopIteration:
			return None


if __name__ == "__main__":
	import sys
	text = open(sys.argv[1], 'r').read()
	lexer = Slex()

	tok = lexer.tokenize(text)

	cnt = 0
	while(1):
		try:
			x = tok.__next__()
			print(x)
			cnt += 1
		except StopIteration:
			print(None)
			break
	print('Token count =', cnt)
	print('Num Lines =', lexer.lineno)




