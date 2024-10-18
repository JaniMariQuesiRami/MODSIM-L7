import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def adaptive_line_search(f, df, current_point, pk, alpha=1, rho=0.5, c=1e-4):
    fk = f(current_point)
    gk = df(current_point)
    while f(current_point + alpha * pk) > fk + c * alpha * np.dot(gk, pk):
        alpha *= rho
    return alpha

def stochastic_gradient_descent(f, df, starting_point, alpha, max_iterations, tolerance, stopping_criterion=None):
    point_history = [starting_point.copy()]
    func_values_history = [f(starting_point)]
    error_history = []
    current_point = starting_point.copy()
    success = False

    for k in range(1, max_iterations + 1):
        gradient = df(current_point)
        
        random_dir = np.random.randn(*current_point.shape)
        random_dir /= np.linalg.norm(random_dir)
        
        while np.dot(random_dir, gradient) >= 0:
            random_dir = np.random.randn(*current_point.shape)
            random_dir /= np.linalg.norm(random_dir)
        
        
        alpha = adaptive_line_search(f, df, current_point, random_dir)
        xk_new = current_point + alpha * random_dir
        point_history.append(xk_new.copy())
        fxk_new = f(xk_new)
        func_values_history.append(fxk_new)
        
        error = np.linalg.norm(xk_new - current_point)
        error_history.append(error)
        
        if stopping_criterion:
            if stopping_criterion(current_point, xk_new, func_values_history[-2], fxk_new, gradient, k):
                success = True
                break
        elif error < tolerance:
            success = True
            break
        current_point = xk_new

    num_iters = len(point_history) - 1
    x_best = point_history[-1]

    return x_best, point_history, func_values_history, error_history, num_iters, success


def classical_gradient_descent(f, df, starting_point, alpha, max_iterations, tolerance, stopping_criterion=None):
    point_history = [starting_point.copy()]
    func_values_history = [f(starting_point)]
    error_history = []
    current_point = starting_point.copy()
    success = False

    for k in range(1, max_iterations + 1):
        gradient = df(current_point)
        
        alpha = adaptive_line_search(f, df, current_point, -gradient)
        xk_new = current_point + alpha * (-gradient)
        point_history.append(xk_new.copy())
        fxk_new = f(xk_new)
        func_values_history.append(fxk_new)
        
        error = np.linalg.norm(xk_new - current_point)
        error_history.append(error)
        
        if stopping_criterion:
            if stopping_criterion(current_point, xk_new, func_values_history[-2], fxk_new, gradient, k):
                success = True
                break
        elif error < tolerance:
            success = True
            break
        current_point = xk_new

    num_iters = len(point_history) - 1
    x_best = point_history[-1]

    return x_best, point_history, func_values_history, error_history, num_iters, success


def estimate_hessian(df, x, epsilon=1e-5):
    n = x.shape[0]
    H = np.zeros((n, n))
    df_x = df(x)
    for i in range(n):
        x_eps = x.copy()
        x_eps[i] += epsilon
        df_x_eps = df(x_eps)
        H[:, i] = (df_x_eps - df_x) / epsilon
    return H


def newton_with_approx_hessian(f, df, starting_point, alpha, max_iterations, tolerance, stopping_criterion=None):
    point_history = [starting_point.copy()]
    func_values_history = [f(starting_point)]
    error_history = []
    current_point = starting_point.copy()
    success = False

    for k in range(1, max_iterations + 1):
        gradient = df(current_point)
        H_approx = estimate_hessian(df, current_point)
        try:
            
            delta_xk = -alpha * np.linalg.solve(H_approx, gradient)
        except np.linalg.LinAlgError:
            delta_xk = -alpha * gradient
        
        alpha = adaptive_line_search(f, df, current_point, delta_xk)
        xk_new = current_point + alpha * delta_xk
        point_history.append(xk_new.copy())
        fxk_new = f(xk_new)
        func_values_history.append(fxk_new)
        
        error = np.linalg.norm(xk_new - current_point)
        error_history.append(error)
        
        if stopping_criterion:
            if stopping_criterion(current_point, xk_new, func_values_history[-2], fxk_new, gradient, k):
                success = True
                break
        elif error < tolerance:
            success = True
            break
        current_point = xk_new

    num_iters = len(point_history) - 1
    x_best = point_history[-1]

    return x_best, point_history, func_values_history, error_history, num_iters, success


def newton_with_exact_hessian(f, df, ddf, starting_point, alpha, max_iterations, tolerance, stopping_criterion=None):
    point_history = [starting_point.copy()]
    func_values_history = [f(starting_point)]
    error_history = []
    current_point = starting_point.copy()
    success = False

    for k in range(1, max_iterations + 1):
        gradient = df(current_point)
        Hessian = ddf(current_point)
        try:
            
            delta_xk = -alpha * np.linalg.solve(Hessian, gradient)
        except np.linalg.LinAlgError:
            
            delta_xk = -alpha * gradient
        
        alpha = adaptive_line_search(f, df, current_point, delta_xk)
        xk_new = current_point + alpha * delta_xk
        point_history.append(xk_new.copy())
        fxk_new = f(xk_new)
        func_values_history.append(fxk_new)
        
        error = np.linalg.norm(xk_new - current_point)
        error_history.append(error)
        
        if stopping_criterion:
            if stopping_criterion(current_point, xk_new, func_values_history[-2], fxk_new, gradient, k):
                success = True
                break
        elif error < tolerance:
            success = True
            break
        current_point = xk_new

    num_iters = len(point_history) - 1
    x_best = point_history[-1]

    return x_best, point_history, func_values_history, error_history, num_iters, success

import numpy as np


def func_a(x):
    return x[0]**4 + x[1]**4 - 4*x[0]*x[1] + 0.5*x[1] + 1

def grad_a(x):
    df_dx0 = 4*x[0]**3 - 4*x[1]
    df_dx1 = 4*x[1]**3 - 4*x[0] + 0.5
    return np.array([df_dx0, df_dx1])

def hessian_a(x):
    ddf_dx0dx0 = 12*x[0]**2
    ddf_dx0dx1 = -4
    ddf_dx1dx0 = -4
    ddf_dx1dx1 = 12*x[1]**2
    return np.array([[ddf_dx0dx0, ddf_dx0dx1], [ddf_dx1dx0, ddf_dx1dx1]])


def func_b(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def grad_b(x):
    df_dx0 = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    df_dx1 = 200 * (x[1] - x[0]**2)
    return np.array([df_dx0, df_dx1])

def hessian_b(x):
    ddf_dx0dx0 = 1200*x[0]**2 - 400*x[1] + 2
    ddf_dx0dx1 = -400*x[0]
    ddf_dx1dx0 = -400*x[0]
    ddf_dx1dx1 = 200
    return np.array([[ddf_dx0dx0, ddf_dx0dx1], [ddf_dx1dx0, ddf_dx1dx1]])


def func_c(x):
    return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1))

def grad_c(x):
    gradient = np.zeros_like(x)
    n = len(x)
    for i in range(n - 1):
        gradient[i] = -400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
        gradient[i+1] += 200 * (x[i+1] - x[i]**2)
    return gradient

def hessian_c(x):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n - 1):
        H[i, i] += 1200 * x[i]**2 - 400 * x[i+1] + 2
        H[i, i+1] = -400 * x[i]
        H[i+1, i] = -400 * x[i]
        H[i+1, i+1] += 200
    return H

def display_results_and_graphs(methods, f, df, ddf, starting_point, max_iterations, tolerance):
    results = []
    for method_name, method_func in methods.items():
        if method_name == "Newton Exact":
            x_best, point_history, func_values_history, error_history, num_iters, success = method_func(f, df, ddf, starting_point, 1, max_iterations, tolerance)
        else:
            x_best, point_history, func_values_history, error_history, num_iters, success = method_func(f, df, starting_point, 1, max_iterations, tolerance)
        
        
        first_three = point_history[:3]
        last_three = point_history[-3:]
        approx_errors = [np.linalg.norm(x - x_best) for x in first_three + last_three]
        grad_norms = [np.linalg.norm(df(x)) for x in first_three + last_three]
        
        table_data = []
        for i, (x, error, grad_norm) in enumerate(zip(first_three + last_three, approx_errors, grad_norms)):
            iteration = i + 1 if i < 3 else num_iters - 2 + i % 3
            table_data.append([method_name, iteration, x, error, grad_norm])
        
        results.append((method_name, error_history, table_data, point_history))
    
    
    all_table_data = [row for _, _, table_data, _ in results for row in table_data]
    print(tabulate(all_table_data, headers=["Method", "Iteration", "x_k", "Approx. Error", "gradient Norm"]))
    
    
    plt.figure(figsize=(10, 6))
    for method_name, error_history, _, _ in results:
        plt.semilogy(range(1, len(error_history) + 1), error_history, label=method_name)
    plt.xlabel("Iteration")
    plt.ylabel("Approximation Error (log scale)")
    plt.title("Comparison of Approximation Errors")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    if len(starting_point) == 2:
        plt.figure(figsize=(10, 8))
        
        
        all_points = [point for _, _, _, point_history in results for point in point_history]
        x_min, x_max = min(p[0] for p in all_points), max(p[0] for p in all_points)
        y_min, y_max = min(p[1] for p in all_points), max(p[1] for p in all_points)
        
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= 0.1 * x_range
        x_max += 0.1 * x_range
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
        
        
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([f(np.array([xi, yi])) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)
        
        plt.contour(X, Y, Z, levels=20, cmaps='viridis')
        plt.colorbar(label='f(x)')
        
        
        for method_name, _, _, point_history in results:
            plt.plot([x[0] for x in point_history], [x[1] for x in point_history], label=method_name)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Path of Points Generated by Algorithms')
        plt.legend()
        plt.show()

methods = {
    "Stochastic Random": stochastic_gradient_descent,
    "Classical Decent": classical_gradient_descent,
    "Newton Approx": newton_with_approx_hessian,
    "Newton Exact": newton_with_exact_hessian
}

starting_point = np.array([-3, 1])
max_iterations = 100
tolerance = 1e-6

display_results_and_graphs(methods, func_a, grad_a, hessian_a, starting_point, max_iterations, tolerance)
