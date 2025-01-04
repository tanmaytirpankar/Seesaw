
from gtokens import *
import symengine as seng
from sly import Lexer

class Slex(Lexer):

	tokens = { INPUTS, OUTPUTS, EXPRS, REQUIRES,		\
				INTEGER, FLOAT, PLUS,  		\
				MINUS, MUL, DIV, SQRT, 		\
				SIN, ASIN, COS, TAN, COT, COSH, 	\
				SINH, LOG, EXP, IDEN, 		\
			   	FPTRUNC, FPEXT,	\
				EQ, NEQ, \
				ASSIGN, LPAREN, RPAREN, 	\
				SLPAREN, SRPAREN, COLON,	\
				SEMICOLON, COMMA, ID, FPTYPE, INTTYPE, \
				IF, THEN, ELSE, ENDIF, AND, \
				OR, NOT, LEQ, LT, GEQ, GT	\
			}

	ignore = ' \t'
	ignore_comment = r'\#.*'

	# regular expressions
	PLUS		=	r'\+'	
	MUL			=	r'\*'	
	DIV			=	r'\/'	
	EQ			=	r'\=='	
	NEQ			=	r'\!='	
	ASSIGN		=	r'\='	
	LPAREN		=	r'\('	
	RPAREN		=	r'\)'	
	SLPAREN		=	r'\{'	
	SRPAREN		=	r'\}'
	COLON		=	r'\:'
	SEMICOLON	=	r'\;'
	COMMA		=	r'\,'
	AND			=	r'\&&'
	OR			=	r'\|\|'
	NOT			=	r'\!'
	LEQ			=	r'\<='
	LT			=	r'\<'
	GEQ			=	r'\>='
	GT			=	r'\>'

	ID				=	r'[a-zA-Z][a-zA-Z0-9_]*'
	ID['INPUTS']	=	INPUTS
	ID['OUTPUTS']	=	OUTPUTS
	ID['EXPRS']		=	EXPRS
	ID['REQUIRES']	=	REQUIRES
	ID['sqrt']		=	SQRT
	ID['sin']		= 	SIN
	ID['asin']		= 	SIN
	ID['cos']		= 	COS
	ID['log']		= 	LOG
	ID['exp']		= 	EXP
	ID['tan']		= 	TAN
	ID['cot']		= 	COT
	ID['cosh']		= 	COSH
	ID['sinh']		= 	SINH
	ID['fptrunc']	= 	FPTRUNC
	ID['fpext']		= 	FPEXT
	ID['rnd16'] 	= 	FPTYPE
	ID['rnd32'] 	= 	FPTYPE # 'rnd64', 'fl16', 'fl32', 'fl64'] = FPTYPE
	ID['rnd64'] 	= 	FPTYPE
	ID['fl16']  	= 	FPTYPE
	ID['fl32']  	= 	FPTYPE
	ID['fl64']  	= 	FPTYPE
	ID['int']		= 	INTTYPE
	ID['if']		=	IF
	ID['then']		=	THEN
	ID['else']		=	ELSE
	ID['endif']		=	ENDIF


	current_token = None
	tok = None


	def ID(self, t):
		if t.type not in  (INPUTS, OUTPUTS, EXPRS, REQUIRES):
			t.value = seng.var(t.value)
		return t

	@_(r'[\-]?\d+\.\d+([eE][-+]?\d+)?')
	def FLOAT(self, t):
		t.value = float(t.value)
		return t

	@_(r'[\-]?\d+')
	def INTEGER(self, t):
		t.value = int(t.value)
		t.type = INTEGER
		return t

	MINUS	=	r'\-'

	@_(r'\n+')
	def ignore_newline(self, t):
		self.lineno += t.value.count('\n')

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




