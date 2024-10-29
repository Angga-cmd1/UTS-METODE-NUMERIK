import numpy as np
import matplotlib.pyplot as plt

# Constants for inductance (L), capacitance (C), and desired resonance frequency
L = 0.5  # Inductance in Henrys
C = 10e-6  # Capacitance in Farads
target_freq = 1000  # Desired frequency in Hz
tolerance = 1e-3  # Precision for methods

# Function to compute resonance frequency for a given resistance value
def calc_resonance_freq(R):
    sqrt_part = 1 / (L * C) - (R ** 2) / (4 * L ** 2)
    if sqrt_part <= 0:
        return None  # Invalid case if square root term is non-positive
    return (1 / (2 * np.pi)) * np.sqrt(sqrt_part)

# Function for derivative of resonance frequency for Newton-Raphson approach
def freq_derivative(R):
    sqrt_part = 1 / (L * C) - (R ** 2) / (4 * L ** 2)
    if sqrt_part <= 0:
        return None
    return -R / (4 * np.pi * L ** 2 * np.sqrt(sqrt_part))

# Newton-Raphson method to identify resistance for target frequency
def newton_raphson_method(initial_R, tolerance):
    R = initial_R
    while True:
        freq_value = calc_resonance_freq(R)
        if freq_value is None:
            return None
        error = freq_value - target_freq
        derivative = freq_derivative(R)
        if derivative is None:
            return None
        new_R = R - error / derivative
        if abs(new_R - R) < tolerance:
            return new_R
        R = new_R

# Bisection method to find resistance within a given range
def bisection_method(a, b, tolerance):
    while (b - a) / 2 > tolerance:
        midpoint = (a + b) / 2
        midpoint_freq_diff = calc_resonance_freq(midpoint) - target_freq
        if midpoint_freq_diff is None:
            return None
        if abs(midpoint_freq_diff) < tolerance:
            return midpoint
        if (calc_resonance_freq(a) - target_freq) * midpoint_freq_diff < 0:
            b = midpoint
        else:
            a = midpoint
    return (a + b) / 2

# Initial setup for resistance finding process
initial_resistance = 50
search_range_start, search_range_end = 0, 100

# Calculate resistance using both methods
resistance_via_newton = newton_raphson_method(initial_resistance, tolerance)
frequency_newton = calc_resonance_freq(resistance_via_newton) if resistance_via_newton is not None else "Not found"
resistance_via_bisection = bisection_method(search_range_start, search_range_end, tolerance)
frequency_bisection = calc_resonance_freq(resistance_via_bisection) if resistance_via_bisection is not None else "Not found"

# Display results
print("Newton-Raphson Method Results:")
print(f"Resistance: {resistance_via_newton} ohm, Resonance Frequency: {frequency_newton} Hz")
print("\nBisection Method Results:")
print(f"Resistance: {resistance_via_bisection} ohm, Resonance Frequency: {frequency_bisection} Hz")

# Plotting results for comparison
plt.figure(figsize=(10, 5))
plt.axhline(target_freq, color="red", linestyle="--", label="Target Frequency 1000 Hz")

if resistance_via_newton is not None:
    plt.scatter(resistance_via_newton, frequency_newton, color="blue", label="Newton-Raphson")
    plt.text(resistance_via_newton, frequency_newton + 30, f"NR: R={resistance_via_newton:.2f}, f={frequency_newton:.2f} Hz", color="blue")

if resistance_via_bisection is not None:
    plt.scatter(resistance_via_bisection, frequency_bisection, color="green", label="Bisection")
    plt.text(resistance_via_bisection, frequency_bisection + 30, f"Bisection: R={resistance_via_bisection:.2f}, f={frequency_bisection:.2f} Hz", color="green")

plt.xlabel("Resistance R (Ohm)")
plt.ylabel("Resonance Frequency f(R) (Hz)")
plt.title("Comparison of Newton-Raphson and Bisection Methods")
plt.legend()
plt.grid(True)
plt.show()

# Define linear system for Gaussian and Gauss-Jordan elimination
A = np.array([[1, 1, 1],
              [1, 2, -1],
              [2, 1, 2]], dtype=float)
b = np.array([6, 2, 10], dtype=float)

# Gaussian elimination function
def gaussian_elimination(A, b):
    n = len(b)
    augmented_matrix = np.hstack([A, b.reshape(-1, 1)])
    for i in range(n):
        for j in range(i + 1, n):
            factor = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j, i:] -= factor * augmented_matrix[i, i:]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i + 1:n], x[i + 1:n])) / augmented_matrix[i, i]
    return x

# Gauss-Jordan elimination function
def gauss_jordan_elimination(A, b):
    n = len(b)
    augmented_matrix = np.hstack([A, b.reshape(-1, 1)])
    for i in range(n):
        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i, i]
        for j in range(n):
            if i != j:
                augmented_matrix[j] -= augmented_matrix[i] * augmented_matrix[j, i]
    return augmented_matrix[:, -1]

# Solve linear system using both methods
solution_gaussian = gaussian_elimination(A, b)
solution_gauss_jordan = gauss_jordan_elimination(A, b)

print("Solution using Gaussian Elimination:")
print(f"x1 = {solution_gaussian[0]}, x2 = {solution_gaussian[1]}, x3 = {solution_gaussian[2]}")
print("\nSolution using Gauss-Jordan Elimination:")
print(f"x1 = {solution_gauss_jordan[0]}, x2 = {solution_gauss_jordan[1]}, x3 = {solution_gauss_jordan[2]}")
