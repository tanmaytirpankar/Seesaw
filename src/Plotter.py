from argparse import ArgumentParser
from sympy import Symbol
from matplotlib import pyplot
from random import uniform
from sympy.utilities import lambdify


class Plotter(object):
    """
    Base Class for generating plots

    Attributes
    ----------
    function :
        Symbolic Expression
    inputs : dict/list
        Dictionary(Symbol -> list[]) of input intervals or a list of points in space
    use_intervals : bool
        Decides whether inputs are intervals or points
    number_of_points : int
        If inputs are intervals, this indicates the number of points to be generated, if inputs are points, it is just
        the number of points
    x_axis_values : list
        Values to be plotted on the x-axis
    y_axis_values : list
        Values to be plotted on the y-axis
    x_axis_label : str
        Label for the x-axis
    y_axis_label : str
        Label for the y-axis
    """

    def __init__(self, symbolic_expression, inputs, use_intervals=False, number_of_points=3):
        """
        Class initializer

        Parameters
        ----------
        symbolic_expression :
            Symbolic Expression
        inputs : dict
            Dictionary(Symbol -> list[]) of input intervals or a list of points in space
        use_intervals : bool
            Decides whether inputs are intervals or points

        Returns
        -------
        None

        Example:
        >>> object1 = Plotter(Symbol('x')+Symbol('y')+Symbol('z'), {Symbol('x'): [0, 1], Symbol('y'): [0, 2],
        ... Symbol('z'): [1, 2]}, use_intervals=True)
        >>> object1.function
        x + y + z
        >>> object1.inputs
        {x: [0, 1], y: [0, 2], z: [1, 2]}
        >>> object1.use_intervals
        True
        >>> assert object1.use_intervals
        >>> assert not (object1.x_axis_values or object1.y_axis_values or object1.x_axis_label or object1.y_axis_label)
        """
        self.inputs = inputs
        self.function = symbolic_expression
        self.use_intervals = use_intervals
        if self.use_intervals:
            assert type(inputs) == dict
            self.number_of_points = number_of_points
        else:
            assert type(inputs) == list
            self.number_of_points = len(inputs)
        self.x_axis_values = []
        self.y_axis_values = []
        self.x_axis_label = ""
        self.y_axis_label = ""

    def __str__(self):
        """
        String representation of this class

        Example:
        >>> print(Plotter(Symbol('x')+Symbol('y')+Symbol('z'), {Symbol('x'): [0, 1], Symbol('y'): [0, 2], Symbol('z'):
        ... [1, 2]}, use_intervals=True, number_of_points=5))
        Inputs  : {x: [0, 1], y: [0, 2], z: [1, 2]}
        Function: x + y + z
        Interval: True
        Points  : 5
        >>> print(Plotter(Symbol('x')+Symbol('y')+Symbol('z'), [{Symbol('x'): 0, Symbol('y'): 0, Symbol('z'): 1},
        ... {Symbol('x'): 1, Symbol('y'): 0, Symbol('z'): 2}, {Symbol('x'): 2, Symbol('y'): 0, Symbol('z'): 3},
        ... {Symbol('x'): 3, Symbol('y'): 0, Symbol('z'): 4}, {Symbol('x'): 4, Symbol('y'): 0, Symbol('z'): 5}]))
        Inputs  :
        {x:                       0, y:                       0, z:                       1, }
        {x:                       1, y:                       0, z:                       2, }
        {x:                       2, y:                       0, z:                       3, }
        {x:                       3, y:                       0, z:                       4, }
        {x:                       4, y:                       0, z:                       5, }
        Function: x + y + z
        Interval: False
        Points  : 5
        """
        returning_str = "Inputs".ljust(8) + ':'

        if self.use_intervals:
            returning_str += ' ' + str(self.inputs) + '\n'
        else:
            returning_str += '\n'
            for point in self.inputs:
                returning_str += '{'
                for key, val in point.items():
                    returning_str += str(key) + ": " + str(val).rjust(23) + ', '
                returning_str += '}\n'

        returning_str += "Function".ljust(8) + ": " + str(self.function) + '\n'
        returning_str += "Interval".ljust(8) + ": " + str(self.use_intervals) + '\n'
        returning_str += "Points".ljust(8) + ": " + str(self.number_of_points)

        return returning_str

    def generate_values(self, variable):
        """
        For a variable var such that var_lower<=var_i<var_upper+2**-53
        Note: Only invoke if inputs are intervals

        Parameters
        ----------
        variable
            Variable to generate values for from its interval

        Example:
        >>> object1 = Plotter(Symbol('x')+Symbol('y')+Symbol('z'), {Symbol('x'): [0, 1], Symbol('y'): [0, 2],
        ... Symbol('z'): [1, 2]}, use_intervals=True)
        >>> variable_values = object1.generate_values(Symbol('x'))
        >>> for variable_value in variable_values:
        ...     assert object1.inputs[Symbol('x')][0] <= variable_value <= object1.inputs[Symbol('x')][1]
        """
        assert self.use_intervals and variable in self.inputs

        return [uniform(self.inputs[variable][0], self.inputs[variable][1] + 2**-53) for _ in
                range(self.number_of_points)]

    def generate_points(self):
        """
        For a set of variables X, selects values for var_i in X such that var_i_lower<=var_i<var_i_upper+2**-53
        Note: Only invoke if inputs are intervals

        Parameters
        ----------
            None

        Returns
        -------
        list
            List of points generated from the input box.

        Examples:
        >>> object1 = Plotter(Symbol('x')+Symbol('y')+Symbol('z'), {Symbol('x'): [0, 1], Symbol('y'): [0, 2],
        ... Symbol('z'): [1, 2]}, use_intervals=True)
        >>> inputs = object1.generate_points()
        >>> for point in inputs:
        ...     for var, val in point.items():
        ...         assert object1.inputs[var][0] <= val <= object1.inputs[var][1]
        """
        assert self.use_intervals

        returning_points_list = []

        for i in range(self.number_of_points):
            returning_points_list.append({})
            for var in self.inputs:
                returning_points_list[-1][var] = uniform(self.inputs[var][0], self.inputs[var][1] + 2**-53)

        return returning_points_list

    def plot(self):
        """
        Generates a plot

        Parameters
        ----------
            None

        Example:
        >>> object1 = Plotter(Symbol('x')+Symbol('y')+Symbol('z'), {Symbol('x'): [0, 1], Symbol('y'): [0, 2],
        ... Symbol('z'): [1, 2]}, use_intervals=True)
        >>> object1.x_axis_values = [1, 2, 3, 4]
        >>> object1.y_axis_values = [3, 4, 5, 5]
        >>> object1.x_axis_label = "Points"
        >>> object1.y_axis_label = "Value"

        # TODO: How would you test a plot?
        # >>> object1.plot()
        """
        assert self.x_axis_values and self.y_axis_values and len(self.x_axis_values) == len(self.y_axis_values)

        pyplot.plot(self.x_axis_values, self.y_axis_values)
        pyplot.xlabel(self.x_axis_label)
        pyplot.ylabel(self.y_axis_label)

        pyplot.show()


class FunctionPlotter(Plotter):
    """
    Class for plotting a function against one of its variables keeping the rest constant

    Attributes
    ----------
    parametrized_variable
        Variable to plot the function against
    """

    def __init__(self, symbolic_expression, inputs, parametrized_variable, use_intervals=False, number_of_points=3):
        """
        Class initializer

        Parameters
        ----------
        parametrized_variable
            Variable to plot the function against

        Example:
        >>> object1 = FunctionPlotter(Symbol('x')+Symbol('y')+Symbol('z'), {Symbol('x'): [0, 1], Symbol('y'): [0, 2],
        ... Symbol('z'): [1, 2]}, Symbol('x'), use_intervals=True)
        >>> object1.function
        x + y + z
        >>> object1.inputs
        {x: [0, 1], y: [0, 2], z: [1, 2]}
        >>> object1.use_intervals
        True
        >>> assert object1.use_intervals
        >>> assert not (object1.x_axis_values or object1.y_axis_values)
        >>> object1.parametrized_variable
        x
        >>> object1.x_axis_label
        'x'
        >>> object1.y_axis_label
        'f(x)'
        """
        super().__init__(symbolic_expression, inputs, use_intervals, number_of_points)
        self.parametrized_variable = parametrized_variable
        self.x_axis_label = str(parametrized_variable)
        self.y_axis_label = "f("+str(parametrized_variable)+")"

    def __str__(self):
        """
        String representation of this class

        Example:
        >>> print(FunctionPlotter(Symbol('x')+Symbol('y')+Symbol('z'), {Symbol('x'): [0, 1], Symbol('y'): [0, 2],
        ... Symbol('z'): [1, 2]}, Symbol('x'), use_intervals=True, number_of_points=5))
        Inputs  : {x: [0, 1], y: [0, 2], z: [1, 2]}
        Function: x + y + z
        Interval: True
        Points  : 5
        Plot Var: x
        >>> print(FunctionPlotter(Symbol('x')+Symbol('y')+Symbol('z'), [{Symbol('x'): 0, Symbol('y'): 0, Symbol('z'): 1}
        ... , {Symbol('x'): 1, Symbol('y'): 0, Symbol('z'): 2}, {Symbol('x'): 2, Symbol('y'): 0, Symbol('z'): 3},
        ... {Symbol('x'): 3, Symbol('y'): 0, Symbol('z'): 4}, {Symbol('x'): 4, Symbol('y'): 0, Symbol('z'): 5}],
        ... Symbol('x')))
        Inputs  :
        {x:                       0, y:                       0, z:                       1, }
        {x:                       1, y:                       0, z:                       2, }
        {x:                       2, y:                       0, z:                       3, }
        {x:                       3, y:                       0, z:                       4, }
        {x:                       4, y:                       0, z:                       5, }
        Function: x + y + z
        Interval: False
        Points  : 5
        Plot Var: x
        """
        returning_str = super().__str__() + '\n'
        returning_str += "Plot Var".ljust(8) + ": " + str(self.parametrized_variable)

        return returning_str

    def plot(self):
        """
        Plots function against a variable

        Parameters
        ----------
        None

        Example:
        >>> object1 = FunctionPlotter(Symbol('x')+Symbol('y')+Symbol('z'), {Symbol('x'): [0, 1], Symbol('y'): [0, 2],
        ... Symbol('z'): [1, 2]}, parametrized_variable=Symbol('x'), use_intervals=True)

        # # TODO: How would you test a plot?
        # >>> object1.plot()
        """
        constant_values = {}
        for var in self.inputs:
            if var != self.parametrized_variable:
                constant_values[var] = uniform(self.inputs[var][0], self.inputs[var][1] + 2 ** -53)

        # self.x_axis_values = super().generate_values(self.parametrized_variable)

        inputs = []
        for i in range(self.number_of_points):
            inputs.append({})
            for var in self.inputs:
                if var == self.parametrized_variable:
                    inputs[-1][var] = uniform(self.inputs[var][0], self.inputs[var][1] + 2 ** -53)
                else:
                    inputs[-1][var] = constant_values[var]

        # Collecting values for plotting
        f = lambdify(self.inputs.keys(), self.function)

        print(type(inputs))
        self.x_axis_values = [point[self.parametrized_variable] for point in inputs]
        self.y_axis_values = [f(*point.values()) for point in inputs]

        # Checking whether the argument count and interval counts matches
        assert f.__code__.co_argcount == len(self.inputs)

        super().plot()
        return


def main():
    parser = ArgumentParser(description="Class for plotting functions and/or its properties.")
    parser.add_argument('-e', "--symbolic_expression",
                        help="An expression",
                        required=True,
                        type=str)
    parser.add_argument('-i', "--inputs",
                        help="Inputs",
                        required=True,
                        type=dict)
    parser.add_argument('-in', "--intervals",
                        help="Are the inputs intervals?",
                        required=True,
                        type=bool, action="store_true")
    parser.add_argument('-p', "--plot_type",
                        help="Selects the type of plot:"
                             " 1: Function plot"
                             " 2: Absolute Error plot"
                             " 3: Relative Error plot",
                        type=int, default=1)

    arguments = parser.parse_args()

    # # TODO: Finish the absolute and relative error plot branches
    # if arguments.plot_type == 1:


if __name__ == "__main__":
    main()
