from RandomTesting import BinaryGuidedRandomTesting
from ops_def import _atomic_condition_ops, _atomic_condition_danger_zones
from collections import defaultdict
from utils import isConst
from PredicatedSymbol import SymTup, Sym
from utils import extract_input_dep, extremum_of_symbolic_expression, binary_search_for_input_set
from sympy import Symbol
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
    stability_checked :
    stability_threshold :
    node_visited :
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

        self.node_visited = defaultdict(set)

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
        print("Beginning generating atomic condition expressions...")

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

        print("Completed generating atomic condition expressions...")

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
            elif not self.determine_stability(child):
                return False

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
                    print("Program is unstable at")
                    print(node)
                    print("The expression is:" + str(node.f_expression))
                    print("Atomic condition: " + str(symbolic_expression))
                    print("Atomic condition number = " + str(extremum))
                    if self.analysis_options.find_ill_conditioning_input:
                        input_interval_dict = defaultdict(list)
                        for input_variable in symbolic_expression.free_symbols:
                            input_interval_dict[str(input_variable)] = Globals.inputVars[input_variable]["INTV"]
                        input_interval_dict = binary_search_for_input_set(symbolic_expression, "<<True>>", "", input_interval_dict, extremum, True)
                        print("Input set triggering this atomic condition number")
                        print(input_interval_dict)
                        print()
                        print("Advice")
                        # TODO: child.fexpression is a SymTup so has more than one expression. We select the 1st
                        #  expression right now but it is not correct. Selection of the correct expression is important
                        #  for this "Advice". Change this so the correct expression is selected.

                        constraint = _atomic_condition_danger_zones[node.token.type](
                            [child.f_expression[0].exprCond[0] for child in node.children])
                        if len(constraint) == 0:
                            print("No danger zone")
                        elif len(constraint) == 1 and (constraint[0] != True or constraint[0] != False):
                            print(constraint[0])
                        elif len(constraint) == 2 and (constraint[0] != True or constraint[0] != False) and \
                            (constraint[1] != True or constraint[1] != False):
                            print(str(constraint[0]) + " and " + str(constraint[1]))
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
        print("Beginning stability checking...")

        for node in self.candidate_node_list:
            if self.stability_checked[node.depth].__contains__(node):
                pass
            elif isConst(node):
                self.stability_checked[node.depth].add(node)
            else:
                print("-----------------------------------------------------------------------------------------")
                print("----------------------------- Stability Determination Start -----------------------------")
                print(":")
                if not self.determine_stability(node):
                    print("Following node IS NOT stable \n" + str(node))
                else:
                    print("Following node IS stable \n" + str(node))
                print("------------------------------ Stability Determination End ------------------------------")
                print("-----------------------------------------------------------------------------------------")
                print()

        print("Completed stability checking...")

    def generate_stability_damaging_constraints(self, node):
        """
        Generates constraints from denominators of atomic conditions that could cause cancellations leading to high
        atomic condition numbers.

         Parameters
        ----------
        node : node type
            Any node

        Returns
        -------
        Nothing
            Populates Globals.stability_damaging_constraints
        """
        for child in node.children:
            if self.node_visited[child.depth].__contains__(child):
                pass
            elif len(child.children) == 0:
                self.node_visited[child.depth].add(child)
            else:
                self.generate_stability_damaging_constraints(child)

        constraint = _atomic_condition_danger_zones[node.token.type](
            [child.f_expression[0].exprCond[0] for child in node.children])
        if len(constraint) == 1:
            if constraint[0] != True or constraint[0] != False:
                Globals.stability_damaging_constraints.append(constraint[0])
        elif len(constraint) == 2:
            if constraint[0] != True or constraint[0] != False:
                Globals.stability_damaging_constraints.append(constraint[0])
            if constraint[1] != True or constraint[0] != False:
                Globals.stability_damaging_constraints.append(constraint[1])
        return

    def generate_stability_damaging_constraints_driver(self):
        """
        Driver for generating stability damaging constraints

        Parameters
        ----------
        None

        Returns
        -------
        Nothing
            Populates Globals.stability_damaging_constraints
        """

        for node in self.candidate_node_list:
            if self.node_visited[node.depth].__contains__(node):
                pass
            elif isConst(node):
                self.node_visited[node.depth].add(node)
            else:
                self.generate_stability_damaging_constraints(node)

        # Printing stability damaging constraints
        # print("Stability damaging zones")
        # for i in range(len(Globals.stability_damaging_constraints)):
        #     print(Globals.stability_damaging_constraints[i])

        print("----------------------------------------------------------------------------------------")
        print("---------------------- Search for Stability Damaging Inputs Start ----------------------")

        # Looping through all collected Constraints
        for i in range(len(Globals.stability_damaging_constraints)):
            # Collecting data to invoke Binary Guided Random Testing
            input_configuration = {}
            for key in (Globals.stability_damaging_constraints[i].lhs-Globals.stability_damaging_constraints[i].rhs).free_symbols:
                input_configuration[key] = Globals.inputVars[symengine.var(str(key))]['INTV']
                if Symbol('n') in (Globals.stability_damaging_constraints[i].lhs-Globals.stability_damaging_constraints[i].rhs).free_symbols:
                    input_configuration[Symbol('n')] = [1.0, 1.0]
            symbolic_expression = (Globals.stability_damaging_constraints[i].lhs-Globals.stability_damaging_constraints[i].rhs)

            # Invoking BGRT
            bad_inputs_finder = BinaryGuidedRandomTesting(input_configuration, symbolic_expression,
                                                          configuration_generation_factor=3)
            bad_input_boxes, best_values = bad_inputs_finder.binary_guided_random_testing()

            # Printing the output
            assert len(bad_input_boxes) == len(best_values)
            print("Symbolic Expression:" + str(Globals.stability_damaging_constraints[i].lhs -
                                               Globals.stability_damaging_constraints[i].rhs))
            for j in range(len(bad_input_boxes)):
                print(str(j).rjust(3) + ": value: " + str(best_values[j]).rjust(23) + ", [", end='')
                for key, val in bad_input_boxes[j].items():
                    print(str(key) + ": [" + str(val[0]).ljust(23) + ", " + str(val[1]).ljust(23) + "], ", end='')
                print(']')
            print()

        print("----------------------- Search for Stability Damaging Inputs End -----------------------")
        print("----------------------------------------------------------------------------------------")
        print()
