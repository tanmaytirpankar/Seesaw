import copy

from sympy import Symbol, sin
from sympy.utilities import lambdify
from random import uniform, random
from sys import float_info
from copy import deepcopy

class RandomTesting(object):
    """
    Class for randomized testing.

    Attributes
    ----------
    input_intervals : dict
            Dictionary(Symbol -> list[]) of input intervals.
    symbolic_expression :
        Symbolic Expression
    target_value : float
        Output value of the symbolic expression for which we need to find inputs.
    distant_value : float
        Value supposedly furthest away from target value
    sampling_factor: int
        Number of times to sample from the input interval and evaluate.
    termination_criteria_iterations : int
        Number of iterations as the termination criteria of for this algorithm
    configuration_generation_factor : int
        Number of times to partition and generate new configurations. (Approximately at most 3 * number of input
        variables. No point keeping this too high due to redundant configuration possibility)

    Methods
    -------

    Notes
    -----
    It is assumed that the parameters to function and the input variables match in terms of the count as well as names.
    """

    def __init__(self, input_intervals, symbolic_expression, target_value, distant_value=float_info.max,
                 sampling_factor=10, termination_criteria_iterations=1000, configuration_generation_factor=2):
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
        sampling_factor: int
            Number of times to sample from the input interval and evaluate.
        termination_criteria_iterations : int
            Number of iterations as the termination criteria of for this algorithm
        configuration_generation_factor : int
            Number of times to partition and generate new configurations. (Approximately at most 3 * number of input
            variables. No point keeping this too high due to redundant configuration possibility)

        Returns
        -------
        None

        Example:
        >>> object1 = RandomTesting({Symbol('x'): [1,2]}, sin(Symbol('x'))/Symbol('x'), 0, 10, 10, 1000, 2)
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
        1000
        >>> object1.configuration_generation_factor
        2
        """
        self.initial_configuration = input_intervals
        self.function = symbolic_expression
        self.target_value = target_value
        self.distant_value = distant_value
        self.sampling_factor = sampling_factor
        self.termination_criteria_iterations = termination_criteria_iterations
        self.configuration_generation_factor = configuration_generation_factor

    def __str__(self):
        """
        String representation of this class

        Example:
        >>> print(RandomTesting({Symbol('x'): [1,2], Symbol('y'): [2,3]}, sin(Symbol('x'))/Symbol('x'), target_value=0,
        ... sampling_factor=10))
        Random Testing:
        Inputs  : {x: [1, 2], y: [2, 3]}
        Function: sin(x)/x
        Target  : 0
        Distant : 1.7976931348623157e+308
        Sampling: 10
        Iters   : 1000
        New Conf: 2
        """
        returning_str = "Random Testing:\n"

        returning_str += "Inputs".ljust(8) + ": " + str(self.initial_configuration) + '\n'
        returning_str += "Function".ljust(8) + ": " + str(self.function) + '\n'
        returning_str += "Target".ljust(8) + ": " + str(self.target_value) + '\n'
        returning_str += "Distant".ljust(8) + ": " + str(self.distant_value) + '\n'
        returning_str += "Sampling".ljust(8) + ": " + str(self.sampling_factor) + '\n'
        returning_str += "Iters".ljust(8) + ": " + str(self.termination_criteria_iterations) + '\n'
        returning_str += "New Conf".ljust(8) + ": " + str(self.configuration_generation_factor)

        return returning_str

    def binary_guided_random_testing(self):
        """
        Gives the narrowest box from the given inputs for which symbolic expression gives value closest to
        target_value
        Unguided Random Testing method as implemented in "Efficient Search for Inputs Causing High Floating-point
        Errors"

        Parameters
        ----------
        None

        Returns
        -------
        list
            A list of boxes for which symbolic expression gives value closest to target_value

        Example:
        >>> object1 = RandomTesting({Symbol('x'): [1,2]}, sin(Symbol('x'))/Symbol('x'), 0, 10)
        >>> object1.binary_guided_random_testing()

        """
        target_input_interval_list = []


        for i in range(self.termination_criteria_iterations):


        return target_input_interval_list

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
        >>> object1 = RandomTesting({Symbol('x'): [1,2], Symbol('y'): [2,3]}, sin(Symbol('x'))/Symbol('x'), 0, 10)
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
                new_configurations.append({**c_x_upper, **c_y_lower})
            if {**c_y_upper, **c_x_lower} not in new_configurations:
                new_configurations.append({**c_y_upper, **c_x_lower})

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
        >>> object1 = RandomTesting({Symbol('x'): [1,2]}, sin(Symbol('x'))/Symbol('x'), 0, 10)
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
        Evaluates the symbolic expression with given values and returns result

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
        >>> object1 = RandomTesting({Symbol('x'): [1,2]}, sin(Symbol('x'))/Symbol('x'), 0, 10)
        >>> assert type(object1.evaluate(object1.initial_configuration)) == numpy.float64
        """
        # Creating the lambda function out of the variables and symbolic expression
        # NOTE: It is assumed that the parameters to function and the input variables match in terms of the count as
        # well as names.
        lambda_function = lambdify(input_intervals.keys(), self.function)

        # Checking whether the argument count and interval counts matches
        assert lambda_function.__code__.co_argcount == len(input_intervals)

        # Random sampling once
        # Generating random values from input intervals
        parameter_values = [uniform(interval[0], interval[1]) for var, interval in input_intervals.items()]
        # Calling lambda function using parameters by unpacking above data structure using *
        new_value = lambda_function(*parameter_values)
        final_difference = abs(new_value-self.target_value)

        # Random sampling sampling_factor-1 more times
        for i in range(1, self.sampling_factor):
            parameter_values = [uniform(interval[0], interval[1]) for var, interval in input_intervals.items()]
            new_value = lambda_function(*parameter_values)
            if abs(new_value-self.target_value) < final_difference:
                final_difference = abs(new_value-self.target_value)

        return final_difference
