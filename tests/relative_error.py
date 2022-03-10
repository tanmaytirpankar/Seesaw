from mpmath import *
import numpy as np
import matplotlib.pyplot as plt


def xy_plot(x_arr, y_arr):
    plt.plot(x_arr, y_arr)
    plt.show()
    return


def main():
    # mpf precision parameters
    mp.prec = 112
    mp.dps = 25

    # Function parameters
    # Constant value of the variable to be kept constant
    constant_value = 1

    # Interval value for the other variable (Changes by a factor of interval_multiplier). This value unused
    interval = 0.0000000001
    interval_multiplier = 10

    # 0 - First operand, 1 - Second operand
    value_index = 0

    # Number of values to consider.
    steps = 10

    vals = []
    for num in range(steps):
        vals.append(constant_value + interval * interval_multiplier)
        interval *= interval_multiplier

    abs_err_vals = []
    rel_err_vals = []
    for val in vals:
        if not value_index:
            z_real = mpf(str(val)) - mpf(str(constant_value))
            z_appr = mpf(str(val - constant_value))
        elif value_index:
            z_real = mpf(str(constant_value)) - mpf(str(val))
            z_appr = mpf(str(constant_value - val))
        abs_err_vals.append(abs(z_real - z_appr))
        rel_err_vals.append(abs((z_real - z_appr) / z_real))

    # print("x = ", repr(mpf(str(x))))
    # print("y = ", repr(mpf(str(y))))
    # print()
    # print("z_real = ", repr(z_real))
    # print("z_appr = ", repr(z_appr))
    # print("Abs err = ", abs(z_real-z_appr))
    # print("Rel err = ", abs((z_real - z_appr)/z_real))

    # Linspace includes last point. Use endpoint=False argument to exclude last point.
    x_axis_vals = [log(val-constant_value, interval_multiplier) for val in vals]
    y_axis_vals = rel_err_vals

    print(x_axis_vals)
    print(y_axis_vals)
    assert len(x_axis_vals) == len(y_axis_vals)
    xy_plot(x_axis_vals, y_axis_vals)


if __name__ == "__main__":
    main()
