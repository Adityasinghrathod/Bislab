import numpy as np
import matplotlib.pyplot as plt

def objective_function(x):
    return np.sum(x**2)

num_wolves = 5
num_dimensions = 2
num_iterations = 30
lb = -10
ub = 10

wolves = np.random.uniform(lb, ub, (num_wolves, num_dimensions))

alpha_pos = np.zeros(num_dimensions)
beta_pos = np.zeros(num_dimensions)
delta_pos = np.zeros(num_dimensions)

alpha_score = float('inf')
beta_score = float('inf')
delta_score = float('inf')

alpha_score_history = []

def evaluate_fitness():
    global alpha_pos, beta_pos, delta_pos, alpha_score, beta_score, delta_score

    for wolf in wolves:
        fitness = objective_function(wolf)

        if fitness < alpha_score:
            delta_score = beta_score
            delta_pos = beta_pos.copy()

            beta_score = alpha_score
            beta_pos = alpha_pos.copy()

            alpha_score = fitness
            alpha_pos = wolf.copy()

        elif fitness < beta_score:
            delta_score = beta_score
            delta_pos = beta_pos.copy()

            beta_score = fitness
            beta_pos = wolf.copy()

        elif fitness < delta_score:
            delta_score = fitness
            delta_pos = wolf.copy()

def update_positions(iteration):
    a = 2 - iteration * (2 / num_iterations)

    for i in range(num_wolves):
        for j in range(num_dimensions):
            r1 = np.random.random()
            r2 = np.random.random()

            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * alpha_pos[j] - wolves[i, j])
            X1 = alpha_pos[j] - A1 * D_alpha

            r1 = np.random.random()
            r2 = np.random.random()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * beta_pos[j] - wolves[i, j])
            X2 = beta_pos[j] - A2 * D_beta

            r1 = np.random.random()
            r2 = np.random.random()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * delta_pos[j] - wolves[i, j])
            X3 = delta_pos[j] - A3 * D_delta

            wolves[i, j] = (X1 + X2 + X3) / 3
            wolves[i, j] = np.clip(wolves[i, j], lb, ub)

for iteration in range(num_iterations):
    evaluate_fitness()
    update_positions(iteration)

    alpha_score_history.append(alpha_score)

    print(f"Iteration {iteration+1}/{num_iterations}, Alpha Score: {alpha_score}")

print("Best Solution:", alpha_pos)
print("Best Solution Fitness:", alpha_score)

plt.plot(alpha_score_history)
plt.title('Convergence of Grey Wolf Optimizer')
plt.xlabel('Iteration')
plt.ylabel('Alpha Fitness Score')
plt.grid(True)
plt.show()
