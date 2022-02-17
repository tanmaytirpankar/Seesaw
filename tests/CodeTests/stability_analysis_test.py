import sys

sys.path.insert(1, '../../src')
from  seesaw import create_parser
from StabilityAnalysis import StabilityAnalysis
from parser import Sparser
from ASTtypes import *
import Globals


class TestStabilityAnalysis:
    def test_initialization(self):

        # Resetting this Global variable as tests executed before change it
        Globals.scopeID = -1

        # Preparing to launch code
        from lexer import Slex
        cli_parser = create_parser()
        program_argument_list = cli_parser.parse_args(['--file', 'test_code.txt', '-s'])
        assert program_argument_list.file == 'test_code.txt'
        assert program_argument_list.examine_stability
        enable_constr = program_argument_list.enable_constr

        text = open(program_argument_list.file, 'r').read()
        lexer = Slex()
        parser = Sparser(lexer)
        parser.parse(text)
        del parser
        del lexer

        # Creating object to test
        output_variable_node_list = [predicated_node_tuple[0][0] for predicated_node_tuple in
                                     [Globals.global_symbol_table[0]._symTab[outVar] for outVar in Globals.outVars]]
        stability_analyzer = StabilityAnalysis(program_argument_list, output_variable_node_list)

        # Testing initialization
        # Testing analysis options
        assert stability_analyzer.analysis_options.file == 'test_code.txt'
        assert stability_analyzer.analysis_options.examine_stability == True

        # Testing output node
        assert isinstance(stability_analyzer.candidate_node_list[0], BinOp)

        # Testing for empty atomic_condition_number dictionary
        assert not stability_analyzer.atomic_condition_numbers

        # Testing parent_dict
        assert len(stability_analyzer.parent_dict) == 5

        # Testing empty cond_syms
        assert not stability_analyzer.cond_syms