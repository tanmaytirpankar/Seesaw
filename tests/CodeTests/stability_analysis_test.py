import sys

sys.path.insert(1, '../../src')
from  seesaw import create_parser, perform_stability_analysis
from parser import Sparser
import Globals

class TestStabilityAnalysis:
    def test_initialization(self):
        Globals.scopeID = -1
        from lexer import Slex
        cli_parser = create_parser()
        program_argument_list = cli_parser.parse_args(['--file', 'test_code.txt', '-s'])
        assert program_argument_list.file == 'test_code.txt'
        assert program_argument_list.examine_stability
        argList = program_argument_list
        enable_constr = program_argument_list.enable_constr

        text = open(program_argument_list.file, 'r').read()
        lexer = Slex()
        parser = Sparser(lexer)
        parser.parse(text)
        del parser
        del lexer

        perform_stability_analysis(program_argument_list)


        return