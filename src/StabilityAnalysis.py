from ops_def import _atomic_condition_ops
from collections import defaultdict
from utils import isConst
from PredicatedSymbol import SymTup, Sym
from utils import extract_input_dep, extremum_of_symbolic_expression, binary_search_for_input_set
import symengine
import helper
import Globals


class StabilityAnalysis(object):
    """
    Class representing the stability analysis module of Seesaw

    Attributes
    ----------
    analysis_options :
    candidate_node_list :
    input_node_list :
    atomic_condition_numbers :
    parent_dict :
    cond_syms :
    atomic_condition_expression_generated :
    """
    def __init__(self, analysis_options, candidate_node_list, input_node_list):
        self.analysis_options = analysis_options
        self.candidate_node_list = candidate_node_list
        self.input_node_list = input_node_list
        self.atomic_condition_numbers = defaultdict(dict)
        (self.parent_dict, self.cond_syms) = helper.expression_builder_driver(self.candidate_node_list)

        self.atomic_condition_expression_generated = defaultdict(set)
        self.stability_checked = defaultdict(set)
        self.stability_threshold = 1000

    def generate_atomic_conditions(self, node):
        """
        Traverses the "node" subtree in an INORDER manner. Generates the atomic condition number of the operation at
        "node" for its expression with respect to its operands.

        Parameters
        ----------
        node : node type
            Any node for which atomic condition generation is needed.

        Returns
        -------
        Nothing
            By the end, the atomic condition numbers are stored in 'self' object. All nodes that have been visited are
            marked by marking in 'atomic_condition_expression_generated' dictionary.
        """
        for child in node.children:
            if self.atomic_condition_expression_generated[child.depth].__contains__(child):
                pass
            elif len(child.children) == 0:
                self.atomic_condition_expression_generated[child.depth].add(child)
            else:
                self.generate_atomic_conditions(child)

        # Atomic condition expression generation
        self.atomic_condition_numbers[node] = defaultdict(tuple)
        AtomicFunc = _atomic_condition_ops[node.token.type]
        operand_list = [child.f_expression for child in node.children]

        # Debugging print statments
        # print("\nAtomic condition of:")
        # print(node)

        for i, child in enumerate(node.children):
            self.atomic_condition_numbers[node][child] = SymTup((Sym(0.0, Globals.__T__),)) if isConst(child) else \
                AtomicFunc[i](operand_list)

            # Debugging print statments
            # print("With respect to " + str(i))
            # print(child)
            # print("is")
            # print(self.atomic_condition_numbers[node][child])
            # print()

        self.atomic_condition_expression_generated[node.depth].add(node)

    def generate_atomic_conditions_driver(self):
        """
        Driver function for generation of atomic condition expressions of all nodes in the subtress of each output node.

        Parameters
        ----------
        None

        Returns
        -------
        Nothing
            By the end, the recursive DFS traversal of all output nodes is completed.
        """
        print("Beginning generating atomic condition expressions")

        # Debugging print statments
        # print("\nAtomic condition of:")
        # print(node)

        for node in self.candidate_node_list:
            if self.atomic_condition_expression_generated[node.depth].__contains__(node):
                pass
            elif isConst(node):
                self.atomic_condition_expression_generated[node.depth].add(node)
            else:
                self.generate_atomic_conditions(node)

        print("Completed generating atomic condition expressions")

    # For a node
    def determine_stability(self, node):
        """
        Traverses the "node" subtree in an INORDER manner checking stability at each node. Determines if worst case
        atomic condition number in the user provided interval for this node is greater than stability_threshold.
        If the --find_ill_conditioning_input flag is provided, prints the narrowest input box within user_provided
        interval causing the problem.

        Parameters
        ----------
        node : node type
            Any node for which stability checking is needed

        Returns
        -------
        bool
            Returns True if node is "stable" else False
        """
        for child in node.children:
            if self.stability_checked[child.depth].__contains__(child):
                pass
            elif len(child.children) == 0:
                self.stability_checked[child.depth].add(child)
            else:
                self.determine_stability(child)

        # Invoke gelpia optimizer to determine conditioning
        operand_list = [child.f_expression for child in node.children]

        for i, child in enumerate(node.children):
            symbolic_expression = self.atomic_condition_numbers[node][child][0].exprCond[0]

            # Debugging print statments
            # print("With respect to " + str(i))
            # print(child)
            # print("is")
            # print(self.atomic_condition_numbers[node][child])
            # print()

            if isinstance(symbolic_expression, symengine.Basic):
                input_string = extract_input_dep(list(symbolic_expression.free_symbols))
                # print(input_string)
                # exit(0)
                extremum = extremum_of_symbolic_expression(symbolic_expression, "<<True>>", "", input_string, True)

                # Debugging print statments
                # print("The interval of atomic condition number is")
                # print(val)

                if extremum > self.stability_threshold:
                    print("-----------------------------------------------------------------------------------------")
                    print("Main Output:")
                    print("Atomic condition: " + str(symbolic_expression))
                    print("Atomic condition number = " + str(extremum))
                    print("Program is unstable at")
                    print(node)
                    print("The expression is:" + str(node.f_expression))
                    if self.analysis_options.find_ill_conditioning_input:
                        input_interval_dict = defaultdict(list)
                        for input_variable in symbolic_expression.free_symbols:
                            input_interval_dict[str(input_variable)] = Globals.inputVars[input_variable]["INTV"]
                        input_interval_dict = binary_search_for_input_set(symbolic_expression, "<<True>>", "", input_interval_dict, extremum, True)
                        print("Input set triggering this atomic condition number")
                        print(input_interval_dict)
                        print(
                            "-----------------------------------------------------------------------------------------")
                        return False
                    else:
                        print(
                            "-----------------------------------------------------------------------------------------")
                        return False

        self.stability_checked[node.depth].add(node)
        return True


    def determine_stability_driver(self):
        """
        Driver function for generation of atomic condition expressions of all nodes in the subtress of each output node.

        Parameters
        ----------
        None

        Returns
        -------
        Nothing
            By the end, the recursive DFS traversal of all output nodes is completed.
        """
        print("Beginning stability checking")

        for node in self.candidate_node_list:
            if self.stability_checked[node.depth].__contains__(node):
                pass
            elif isConst(node):
                self.stability_checked[node.depth].add(node)
            else:
                if self.determine_stability(node):
                    print("-----------------------------------------------------------------------------------------")
                    print("Main Output:")
                    print("Following node is stable \n" + str(node))
                    print("-----------------------------------------------------------------------------------------")

        print("Completed stability checking")
