import numpy as np

def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

class Particle:
    def __init__(self, dimension, lower_bound, upper_bound):
        self.position = np.random.uniform(lower_bound, upper_bound, dimension)
        self.velocity = np.random.uniform(-1, 1, dimension)
        self.best_position = np.copy(self.position)
        self.best_value = rastrigin(self.position)

    def update_velocity(self, global_best_position, w, c1, c2):
        r1 = np.random.rand(len(self.position))
        r2 = np.random.rand(len(self.position))

        inertia = w * self.velocity
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)

        self.velocity = inertia + cognitive + social

    def update_position(self, lower_bound, upper_bound):
        self.position = self.position + self.velocity
        self.position = np.clip(self.position, lower_bound, upper_bound)

    def evaluate(self):
        fitness = rastrigin(self.position)

        if fitness < self.best_value:
            self.best_value = fitness
            self.best_position = np.copy(self.position)


def particle_swarm_optimization(dim, lower_bound, upper_bound, num_particles=30,
                                max_iter=100, w=0.5, c1=1.5, c2=1.5):

    particles = [Particle(dim, lower_bound, upper_bound) for _ in range(num_particles)]

    global_best_position = particles[0].best_position.copy()
    global_best_value = particles[0].best_value

    for i in range(max_iter):
        for particle in particles:
            particle.update_velocity(global_best_position, w, c1, c2)
            particle.update_position(lower_bound, upper_bound)
            particle.evaluate()

            if particle.best_value < global_best_value:
                global_best_value = particle.best_value
                global_best_position = np.copy(particle.best_position)

        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}/{max_iter} - Best Fitness: {global_best_value}")

    return global_best_position, global_best_value


dim = 2
lower_bound = -5.12
upper_bound = 5.12
num_particles = 30
max_iter = 100

best_position, best_value = particle_swarm_optimization(
    dim, lower_bound, upper_bound, num_particles, max_iter
)

print("\nBest Solution Found:")
print("Position:", best_position)
print("Fitness:", best_value)
