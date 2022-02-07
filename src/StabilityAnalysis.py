from collections import defaultdict
import helper

class StabilityAnalysis(object):
    def __init__(self, analysis_options, candidate_node_list):
        self.analysis_options = analysis_options
        self.candidate_node_list = candidate_node_list
        self.atomic_condition_numbers = defaultdict(dict)
        (self.parent_dict, self.cond_syms) = helper.expression_builder_driver(self.candidate_node_list)






