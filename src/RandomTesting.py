from sympy import Symbol, sin, parse_expr
from sympy.utilities import lambdify
from random import uniform, random
from sys import float_info
from copy import deepcopy
from argparse import ArgumentParser


class RandomTesting(object):
    """
    Class for randomized testing.

    Attributes
    ----------
    input_intervals : dict
            Dictionary(Symbol -> list[]) of input intervals.
    function :
        Symbolic Expression
    target_value : float
        Output value of the symbolic expression for which we need to find inputs.
    distant_value : float
        Value supposedly furthest away from target value


    Methods
    -------

    Notes
    -----
    It is assumed that the parameters to symbolic expression and the input variables match in terms of the count as
    well as names.
    """

    def __init__(self, input_intervals, symbolic_expression, target_value=0, distant_value=float_info.max):
        """
        Class initializer

        Parameters
        ----------
        input_intervals : dict
            Dictionary(Symbol -> list[]) of input intervals.
        symbolic_expression :
            Symbolic Expression
        target_value : float
            Output value of the symbolic expression for which we need to find inputs.
        distant_value : float
            Value supposedly furthest away from target value

        Returns
        -------
        None

        Example:
        >>> object1 = RandomTesting({Symbol('x'): [1,2]}, sin(Symbol('x'))/Symbol('x'), 0, 10)
        >>> object1.initial_configuration
        {x: [1, 2]}
        >>> object1.function
        sin(x)/x
        >>> object1.target_value
        0
        >>> object1.distant_value
        10
        """
        self.initial_configuration = input_intervals
        self.function = symbolic_expression
        self.target_value = target_value
        self.distant_value = distant_value

    def __str__(self):
        """
        String representation of this class

        Example:
        >>> print(RandomTesting({Symbol('x'): [1,2], Symbol('y'): [2,3]}, sin(Symbol('x'))/Symbol('x'), target_value=0))
        Inputs  : {x: [1, 2], y: [2, 3]}
        Function: sin(x)/x
        Target  : 0
        Distant : 1.7976931348623157e+308
        """
        returning_str = "Inputs".ljust(8) + ": " + str(self.initial_configuration) + '\n'
        returning_str += "Function".ljust(8) + ": " + str(self.function) + '\n'
        returning_str += "Target".ljust(8) + ": " + str(self.target_value) + '\n'
        returning_str += "Distant".ljust(8) + ": " + str(self.distant_value)

        return returning_str


class BinaryGuidedRandomTesting(RandomTesting):
    """
    Class for Binary Guided Randomized Testing

    Attributes
    ----------
    sampling_factor: int
        Number of times to sample from the input interval and evaluate.
    termination_criteria_iterations : int
        Number of iterations as the termination criteria of for this algorithm
    configuration_generation_factor : int
        Number of times to partition and generate new configurations. (Approximately at most 3 * number of input
        variables. No point keeping this too high due to redundant configuration possibility)
    restart_probability : float
        Probability of starting search from initial configuration.

    Methods
    -------
    binary_guided_random_testing
    generate_configurations
    partition_input_box
    evaluate
    print_output

    """
    def __init__(self, input_intervals, symbolic_expression, target_value=0, distant_value=float_info.max,
                 sampling_factor=10, termination_criteria_iterations=100, configuration_generation_factor=2,
                 restart_probability=0.05):
        """
        Class initializer

        Paramters
        ---------
        sampling_factor: int
            Number of times to sample from the input interval and evaluate.
        termination_criteria_iterations : int
            Number of iterations as the termination criteria of for this algorithm
        configuration_generation_factor : int
            Number of times to partition and generate new configurations. (Approximately at most 3 * number of input
            variables. No point keeping this too high due to redundant configuration possibility)
        restart_probability : float
            Probability of starting search from initial configuration. (Set to something less than 0.05 as you dont want
            restarts too often)

        Returns
        -------
        None

        Example:
        >>> object1 = BinaryGuidedRandomTesting({Symbol('x'): [1,2]}, sin(Symbol('x'))/Symbol('x'), 0, 10, 10, 100, 2)
        >>> object1.initial_configuration
        {x: [1, 2]}
        >>> object1.function
        sin(x)/x
        >>> object1.target_value
        0
        >>> object1.distant_value
        10
        >>> object1.sampling_factor
        10
        >>> object1.termination_criteria_iterations
        100
        >>> object1.configuration_generation_factor
        2
        >>> object1.restart_probability
        0.05
        """
        super().__init__(input_intervals, symbolic_expression, target_value, distant_value)
        self.sampling_factor = sampling_factor
        self.termination_criteria_iterations = termination_criteria_iterations
        self.configuration_generation_factor = configuration_generation_factor
        self.restart_probability = restart_probability

    def __str__(self):
        """
        String representation of this class

        Example:
        >>> print(BinaryGuidedRandomTesting({Symbol('x'): [1,2], Symbol('y'): [2,3]}, sin(Symbol('x'))/Symbol('x'),
        ... target_value=0))
        Binary Guided Random Testing:
        Inputs  : {x: [1, 2], y: [2, 3]}
        Function: sin(x)/x
        Target  : 0
        Distant : 1.7976931348623157e+308
        Sampling: 10
        Iters   : 100
        New Conf: 2
        Restart : 0.05
        """
        returning_str = "Binary Guided Random Testing:\n"
        returning_str += super().__str__() + '\n'
        returning_str += "Sampling".ljust(8) + ": " + str(self.sampling_factor) + '\n'
        returning_str += "Iters".ljust(8) + ": " + str(self.termination_criteria_iterations) + '\n'
        returning_str += "New Conf".ljust(8) + ": " + str(self.configuration_generation_factor) + '\n'
        returning_str += "Restart".ljust(8) + ": " + str(self.restart_probability)

        return returning_str

    def binary_guided_random_testing(self):
        """
        Gives the narrowest box and corresponding best values found from the given inputs for which symbolic expression
        gives value closest to target_value
        Unguided Random Testing method as implemented in "Efficient Search for Inputs Causing High Floating-point
        Errors"

        Returns
        -------
        (list, list)
            A tuple of a list of boxes for which symbolic expression gives best value closest to target_value and
            another list of best values found in those intervals

        Example:
        >>> object1 = BinaryGuidedRandomTesting({Symbol('x'): [1,2]}, sin(Symbol('x'))/Symbol('x'), 0,
        ... termination_criteria_iterations=100)
        >>> result = object1.binary_guided_random_testing()
        >>> assert len(result) == 2
        >>> assert type(result) == tuple
        """
        return_values_list = []
        return_input_interval_list = []

        # Initial Setup
        best_value = self.distant_value
        smallest_difference = abs(self.distant_value-self.target_value)
        best_configuration = self.initial_configuration

        # Looping till termination criteria reached
        for i in range(self.termination_criteria_iterations):
            # Generating new configurations
            new_configurations = self.generate_configurations(best_configuration)

            # Evaluating symbolic expression for each configuration to find the configuration giving closest value to
            # target
            for input_configuration in new_configurations:
                new_value = self.evaluate(input_configuration)

                # If better value found, record it and the corresponding configuration
                if abs(new_value-self.target_value) < smallest_difference:
                    smallest_difference = abs(new_value-self.target_value)
                    best_value = new_value
                    best_configuration = deepcopy(input_configuration)
                    # print(best_configuration)
                    # print(smallest_difference)

            # Restarting with some probability to allow exploring other intervals and not getting stuck in a rabbit
            # hole.
            # Add the best value and configuration to list before restarting
            if random() < self.restart_probability:
                # print("Restarting from initial configuration:")
                return_input_interval_list.append(deepcopy(best_configuration))
                return_values_list.append(best_value)
                best_configuration = deepcopy(self.initial_configuration)
                best_value = self.distant_value

        return_input_interval_list.append(deepcopy(best_configuration))
        return_values_list.append(best_value)
        # print(return_input_interval_list)
        # print(return_values_list)
        return return_input_interval_list, return_values_list

    def generate_configurations(self, input_configuration):
        """
        Generates input configurations by dividing input set, splitting into upper and lower and permuting intervals.

        Parameters
        ----------
        input_configuration : dict
            Dictionary(Symbol -> list[]) of input intervals.

        Returns
        -------
        list
            A list of new input configurations

        Example:
        >>> object1 = BinaryGuidedRandomTesting({Symbol('x'): [1,2], Symbol('y'): [2,3]}, sin(Symbol('x'))/Symbol('x'), 0, 10)
        >>> assert len(object1.generate_configurations(object1.initial_configuration)) > 1
        """
        new_configurations = []

        # Subdividing original configuration into upper and lower halves and adding as candidates
        input_configuration_upper = deepcopy(input_configuration)
        input_configuration_lower = deepcopy(input_configuration)
        for key in input_configuration.keys():
            input_configuration_upper[key][0] = (input_configuration_upper[key][0]+input_configuration_upper[key][1])/2
            input_configuration_lower[key][1] = (input_configuration_lower[key][0]+input_configuration_lower[key][1])/2
        new_configurations.append(input_configuration_upper)
        new_configurations.append(input_configuration_lower)

        # Partitioning original configuration, splitting and permuting intervals
        for i in range(self.configuration_generation_factor):
            c_x, c_y = self.partition_input_box(input_configuration)
            c_x_upper = deepcopy(c_x)
            c_x_lower = deepcopy(c_x)
            c_y_upper = deepcopy(c_y)
            c_y_lower = deepcopy(c_y)
            for key in c_x.keys():
                c_x_upper[key][0] = (c_x_upper[key][0] + c_x_upper[key][1])/2
                c_x_lower[key][1] = (c_x_lower[key][0] + c_x_lower[key][1])/2
            for key in c_y.keys():
                c_y_upper[key][0] = (c_y_upper[key][0] + c_y_upper[key][1])/2
                c_y_lower[key][1] = (c_y_lower[key][0] + c_y_lower[key][1])/2

            if {**c_x_upper, **c_y_lower} not in new_configurations:
                new_configurations.append({})

                # Alternative 1
                # Looping and adding one by one instead of unpacking directly to preserve ordering of input_dictionary.
                for key in input_configuration:
                    if key in c_x_upper:
                        new_configurations[-1][key] = c_x_upper[key]
                    elif key in c_y_lower:
                        new_configurations[-1][key] = c_y_lower[key]

                # Alternative 2
                # Unpacking directly; Disrupts ordering of items from input_dictionary
                # new_configurations.append({**c_x_upper, **c_y_lower})
            if {**c_y_upper, **c_x_lower} not in new_configurations:
                # Alternative 1
                # Looping and adding one by one instead of unpacking directly to preserve ordering of input_dictionary.
                for key in input_configuration:
                    if key in c_y_upper:
                        new_configurations[-1][key] = c_y_upper[key]
                    elif key in c_x_lower:
                        new_configurations[-1][key] = c_x_lower[key]

                # Alternative 2
                # Unpacking directly; Disrupts ordering of items from input_dictionary
                # new_configurations.append({**c_y_upper, **c_x_lower})

        return new_configurations

    def partition_input_box(self, input_configuration):
        """
        Partitions input box into two sets of variables

        Parameters
        ----------
        input_configuration
            Dictionary(Symbol -> list[]) of input intervals.

        Returns
        -------
        tuple
            A tuple of two dictionaries c_x and c_y the union of which is the initial_configuration dictionary

        Example:
        >>> object1 = BinaryGuidedRandomTesting({Symbol('x'): [1,2]}, sin(Symbol('x'))/Symbol('x'), 0)
        >>> assert len(object1.partition_input_box({Symbol('x'): [1,2]})) == 2
        """
        c_x = {}
        c_y = {}
        for key, val in input_configuration.items():
            if random() >= 0.5:
                c_x[key] = deepcopy(val)
            else:
                c_y[key] = deepcopy(val)

        return c_x, c_y

    def evaluate(self, input_intervals):
        """
        Evaluates the symbolic expression by drawing inputs from the given input intervals randomly and returns result
        closest to target. The inputs are drawn from the half-open interval [low, high + 2**-53)
        For a set of variables X, selects inputs for var_i in X such that var_i_lower<=var_i<var_i_upper+2**-53

        Parameters
        ----------
        input_intervals : dict
            Dictionary(Symbol -> list[]) of input intervals.

        Returns
        -------
        float
            Returns the closest value to target value obtained by random sampling

        Example
        -------
        >>> import numpy
        >>> object1 = BinaryGuidedRandomTesting({Symbol('x'): [1,2]}, sin(Symbol('x'))/Symbol('x'), 0)
        >>> assert type(object1.evaluate(object1.initial_configuration)) == numpy.float64
        """
        # Creating the lambda function out of the variables and symbolic expression
        # NOTE: It is assumed that the parameters to function and the input variables match in terms of the count as
        # well as names.
        lambda_function = lambdify(input_intervals.keys(), self.function)

        # Checking whether the argument count and interval counts matches
        assert lambda_function.__code__.co_argcount == len(input_intervals)

        # Calling lambda function using parameters by unpacking above data structure using *
        best_value = self.distant_value
        smallest_difference = abs(self.distant_value - self.target_value)
        # print(final_difference)

        # Random sampling sampling_factor-1 more times
        for i in range(self.sampling_factor):
            # Values are drawn from the half-open interval [low, high + 2**-53)
            parameter_values = [uniform(interval[0], interval[1] + 2**-53) for var, interval in input_intervals.items()]
            new_value = lambda_function(*parameter_values)
            if abs(new_value-self.target_value) < smallest_difference:
                smallest_difference = abs(new_value-self.target_value)
                best_value = new_value
                # print(final_difference)

        return best_value

    def print_output(self, narrowed_inputs, best_values_found):
        """
        Prints the output of BGRT.

        Parameters
        ----------
        narrowed_inputs : list
            List of input configurations that may or may not be better than initial input configuration.
        best_values_found : list
            List of best values found for the given function on random sampling from the corresponding configuration
            from list of input configurations

        Returns
        -------
            Nothing
        """
        assert len(narrowed_inputs) == len(best_values_found)
        print("Symbolic Expression:" + str(self.function))
        for j in range(len(narrowed_inputs)):
            print(str(j).rjust(3) + ": value: " + str(best_values_found[j]).rjust(23) + ", [", end='')
            for key, val in narrowed_inputs[j].items():
                print(str(key) + ": [" + str(val[0]).ljust(23) + ", " + str(val[1]).ljust(23) + "], ", end='')
            print(']')
        print()


def main():
    parser = ArgumentParser(description="Searching for input intervals of a symbolic expression close to some target. "
                                        "Note: The input interval obtained may not necessarily contain the target value")
    parser.add_argument('-e',  "--symbolic_expression",
                        help="An expression",
                        required=True,
                        type=str)
    parser.add_argument('-i', "--input_intervals",
                        help="Input variable intervals",
                        required=True,
                        type=dict)
    parser.add_argument('-t', "--target_value",
                        help="Target value for the expression",
                        type=float, default=0)
    parser.add_argument('-d', '--distant_value',
                        help="Value away from the target value",
                        type=float, default=float_info.max)
    parser.add_argument('-b', "--bgrt",
                        help="Run Binary Guided Random Testing",
                        type=bool, default=False, action="store_true")
    parser.add_argument('-s', "--sampling_factor",
                        help="Number of times to sample for each interval. More is better but increases execution "
                             "time.",
                        type=int, default=10)
    parser.add_argument('-te', "--termination_factor_value",
                        help="Value of the termination factor:"
                             "- For Iterations, its the number of iterations",
                        type=float, default=100)
    parser.add_argument('-c', "--configuration_generation_factor",
                        help="Number of new configurations to generate",
                        type=int, default=2)
    parser.add_argument('-r', "--restart_probability",
                        help="Probability of restarting search from initial configuration",
                        type=float, default=0.05)
    arguments = parser.parse_args()

    # Gathering inputs for random testing
    input_configuration = arguments.input_intervals
    for value in input_configuration.values():
        assert len(value) == 2
    symbolic_expression = parse_expr(arguments.symbolic_expression)
    for variable in symbolic_expression.free_symbols:
        assert Symbol(variable) in input_configuration

    # Search-algorithm selector
    if arguments.bgrt:
        bgrt_object = BinaryGuidedRandomTesting(input_configuration, symbolic_expression, arguments.target_value,
                                                arguments.distant_value, arguments.sampling_factor,
                                                arguments.termination_factor_value,
                                                arguments.configuration_generation_factor,
                                                arguments.restart_probability)

        narrowed_inputs, best_values_found = bgrt_object.binary_guided_random_testing()
        bgrt_object.print_output(narrowed_inputs, best_values_found)


if __name__ == "__main__":
    main()
