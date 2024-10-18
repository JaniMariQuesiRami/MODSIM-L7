import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.cm as cm

# Parámetros globales
k = 8  # Número de gaussianas
sigma = 1.0  # Escala de las gaussianas

# 1. Generar puntos aleatorios en el rectángulo [0, 8] x [0, 8]
np.random.seed(42)  # Semilla para reproducibilidad
x_points = np.random.uniform(0, 8, (k, 2))

# 2. Definir la función suma de gaussianas
def gaussian_sum(x):
    total_sum = 0
    for i in range(k):
        distance = np.linalg.norm(x - x_points[i])**2  # Distancia euclidiana al cuadrado
        total_sum += np.exp(-distance / (2 * sigma))
    return -total_sum  # Negativo para encontrar los mínimos locales

# 3. Aplicar el método de optimización desde varios puntos iniciales
initial_guesses = np.random.uniform(0, 8, (20, 2))  # 20 inicializaciones aleatorias
solutions = []
paths = []

for guess in initial_guesses:
    # Guardar la trayectoria de las iteraciones
    trajectory = []

    def callback(xk):
        trajectory.append(np.copy(xk))

    # Minimización con método BFGS
    res = minimize(gaussian_sum, guess, method='BFGS', callback=callback)
    solutions.append(res.x)
    paths.append(trajectory)

# 4. Visualización de los resultados

# Crear una cuadrícula para el gráfico de contornos
x_grid = np.linspace(0, 8, 200)
y_grid = np.linspace(0, 8, 200)
X, Y = np.meshgrid(x_grid, y_grid)
Z = np.zeros_like(X)

# Calcular los valores de la función en la cuadrícula
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = gaussian_sum([X[i, j], Y[i, j]])

# Graficar el contorno de la función
plt.figure(figsize=(10, 10))
plt.contour(X, Y, Z, levels=40, cmap='viridis')  # Aumentar niveles de contorno

# Graficar las trayectorias de las optimizaciones con colores diferentes
colors = cm.rainbow(np.linspace(0, 1, len(paths)))  # Colores diferentes
for i, (path, color) in enumerate(zip(paths, colors)):
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], '-', color=color, label=f'Iteraciones {i+1}')
    plt.plot(path[-1, 0], path[-1, 1], 'o', color=color)  # Solución final

# Graficar los puntos generados aleatoriamente
plt.plot(x_points[:, 0], x_points[:, 1], 'bo', label='Centros de las gaussianas')

# Mejorar la leyenda
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Mover la leyenda fuera de la gráfica
plt.title('Suma de Gaussianas - Convergencia a mínimos locales con 20 iteraciones')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
