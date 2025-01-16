import matplotlib.pyplot as plt

# Data from the table
iterations = list(range(1, 21))
# single_precision_alpha = [0.1274, 0.0792, 0.0256, 0.0311, 0.0419, 0.0608, 0.0943, 0.152, 0.236, 0.282, 0.142, 0.143, 0.162, 0.106, 0.0627, 0.0377, 0.0233, 0.0151, 0.00993, 0.00654]
# single_precision_beta = [0.21, 0.629, 1.14, 1.36, 1.35, 1.37, 1.55, 1.25, 1.13, 1.66, 1.91, 1.68, 1.94, 1.89, 1.87, 1.54, 1.43, 1.85, 1.29, 1.58]
#
# mixed_precision_alpha = [0.138, 0.0866, 0.0288, 0.0351, 0.0472, 0.0685, 0.106, 0.171, 0.265, 0.315, 0.156, 0.16, 0.182, 0.119, 0.0703, 0.0423, 0.0262, 0.0168, 0.011, 0.00724]
# mixed_precision_beta = [0.0243, 0.646, 1.32, 1.51, 1.52, 1.09, 1.28, 1.44, 0.576, 1.89, 1.86, 1.8, 1.96, 2.13, 1.99, 2.0, 1.99, 1.98, 1.96, 1.69]

single_precision_alpha = [0.1274, 0.0866, 0.0288, 0.0351, 0.0472, 0.0685, 0.106, 0.171, 0.236, 0.282, 0.142, 0.143, 0.162, 0.189, 0.193, 0.254, 0.298, 0.342, 0.418, 0.5]
single_precision_beta = [0.21, 0.629, 0.72, 0.91, 0.91, 0.77, 0.95, 0.84, 1.13, 1.89, 1.91, 1.8, 1.96, 2.13, 1.99, 2, 1.99, 1.98, 1.96, 1.69]

mixed_precision_alpha = [0.138, 0.0792, 0.0256, 0.0311, 0.0472, 0.0685, 0.0943, 0.152, 0.236, 0.282, 0.142, 0.143, 0.162, 0.166, 0.173, 0.238, 0.286, 0.332, 0.397, 0.498]
mixed_precision_beta = [0.0243, 0.0646, 0.54, 0.76, 0.75, 0.49, 0.68, 0.65, 0.576, 1.66, 1.86, 1.68, 1.94, 1.89, 1.87, 1.54, 1.43, 1.95, 1.29, 1.58]
# Plotting
plt.figure(figsize=(10, 6))

# Plot for single precision alpha
plt.plot(iterations, single_precision_alpha, label="Alpha (Single Precision)", linestyle='-', color='b')

# Plot for single precision beta
plt.plot(iterations, single_precision_beta, label="Beta (Single Precision)", linestyle='-', color='g')

# Plot for mixed precision alpha
plt.plot(iterations, mixed_precision_alpha, label="Alpha (Mixed Precision)", linestyle='--', color='b')

# Plot for mixed precision beta
plt.plot(iterations, mixed_precision_beta, label="Beta (Mixed Precision)", linestyle='--', color='g')

# Adding labels and title
plt.xlabel("Iteration Numbers")
plt.ylabel("Absolute Error")
plt.title("Absolute Error vs Iteration Numbers")
plt.legend()

# Display the plot
plt.grid(True)
plt.show()