
import sympy
import symengine as seng

from collections import OrderedDict, defaultdict

## Tracks the total number of SMT calls made
solver_calls = 0

## Place holder for the parsed argument list to be made available 
## as a global variable
argList = []

## Satire root directory
ROOT_DIR = ''

## path of realpaver Library file 
LIBFILE = ''


## Global Flag, disable when expand seems to misbehave
simplify = True

## Global flag to control constraint passing
enable_constr = False

## epsilon symbolic variable 
deps = seng.var('eps')


## Input variable's residential dictionary
inputVars = {eps : {"INTV" : [pow(2,53), pow(2,53)], "EPS": [0.0, 0.0]}}
outVars = []

## SymbolTable Hierarchy Tracker
## GS[LEVEL-ID] = { T : SymtabObject,
##                  F : SymtabObject }
GS = dict()


## Record of nodes at each depth
## depthTable[depth=d] = {set of nodes at depth=d}
depthTable = defaultdict(set)


## For expression hashing -- currently disabled
hashBank = OrderedDict()


##-- For the conditional identifiers
# List of condiitonal identifiers
freeConds = []   
# Mapping from identifier to expressions

## Record of condSym and the condtional node pairs
#  predTable[condSym] = cond_node
predTable = dict()
condTable = dict()

## For program trace, track the line-no to program node
#  progTrace[lineno] = node
progTrace = dict() 

condExprBank = dict()

externPredTable = dict()

InstabID = dict()
InstabDict = dict()

garbageCount = 0


## ID variables 
CSID = 0   ## branch identifiers
EID = 0  ## expression ID
gelpiaID = 0
FID = 0
scopeID = -1 # increment , then use

## Symbolic truth values
bvar = sympy.var('bv')
__T__ = (bvar).subs({bvar:True})
__F__ = (bvar).subs({bvar:False})
