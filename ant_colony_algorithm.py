import numpy as np
import matplotlib.pyplot as plt

cities = np.array([
    [0, 0],
    [1, 5],
    [5, 1],
    [6, 4],
    [7, 8]
])

def calculate_distances(cities):
    num_cities = len(cities)
    distances = np.zeros((num_cities, num_cities))

    for i in range(num_cities):
        for j in range(num_cities):
            distances[i][j] = np.linalg.norm(cities[i] - cities[j])

    return distances

distances = calculate_distances(cities)

num_ants = 10
num_cities = len(cities)
alpha = 1.0
beta = 5.0
rho = 0.5
num_iterations = 30
initial_pheromone = 1.0

pheromone = np.ones((num_cities, num_cities)) * initial_pheromone

def heuristic(distances):
    with np.errstate(divide='ignore'):
        return 1 / distances

eta = heuristic(distances)

def choose_next_city(pheromone, eta, visited):
    probs = []

    for j in range(num_cities):
        if j not in visited:
            pheromone_ij = pheromone[visited[-1], j] ** alpha
            heuristic_ij = eta[visited[-1], j] ** beta
            probs.append(pheromone_ij * heuristic_ij)
        else:
            probs.append(0)

    probs = np.array(probs)
    return np.random.choice(range(num_cities), p=probs / probs.sum())

def construct_solution(pheromone, eta):
    tour = [np.random.randint(0, num_cities)]

    while len(tour) < num_cities:
        next_city = choose_next_city(pheromone, eta, tour)
        tour.append(next_city)

    return tour

def update_pheromones(pheromone, all_tours, distances, best_tour):
    pheromone *= (1 - rho)

    for tour in all_tours:
        tour_length = sum([
            distances[tour[i], tour[i + 1]]
            for i in range(-1, num_cities - 1)
        ])

        for i in range(-1, num_cities - 1):
            pheromone[tour[i], tour[i + 1]] += 1.0 / tour_length

    best_length = sum([
        distances[best_tour[i], best_tour[i + 1]]
        for i in range(-1, num_cities - 1)
    ])

    for i in range(-1, num_cities - 1):
        pheromone[best_tour[i], best_tour[i + 1]] += 1.0 / best_length


def run_aco(distances, num_iterations):
    pheromone = np.ones((num_cities, num_cities)) * initial_pheromone
    best_tour = None
    best_length = float('inf')

    for iteration in range(num_iterations):
        all_tours = [
            construct_solution(pheromone, eta)
            for _ in range(num_ants)
        ]

        all_lengths = [
            sum([
                distances[tour[i], tour[i + 1]]
                for i in range(-1, num_cities - 1)
            ])
            for tour in all_tours
        ]

        current_best_length = min(all_lengths)
        current_best_tour = all_tours[all_lengths.index(current_best_length)]

        if current_best_length < best_length:
            best_length = current_best_length
            best_tour = current_best_tour

        update_pheromones(pheromone, all_tours, distances, best_tour)

        print(f"Iteration {iteration + 1}, Best Length: {best_length}")

    return best_tour, best_length


best_tour, best_length = run_aco(distances, num_iterations)

print(f"Best Tour: {best_tour}")
print(f"Best Tour Length: {best_length}")


def plot_route(cities, best_tour):
    plt.figure(figsize=(8, 6))

    for i in range(len(cities)):
        plt.scatter(cities[i][0], cities[i][1], color='red')
        plt.text(cities[i][0], cities[i][1], f"City {i}", fontsize=12)

    tour_cities = np.array(
        [cities[i] for i in best_tour] + [cities[best_tour[0]]]
    )

    plt.plot(
        tour_cities[:, 0],
        tour_cities[:, 1],
        linestyle='-',
        marker='o',
        color='blue'
    )

    plt.title(f"Best Tour (Length: {best_length})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()


plot_route(cities, best_tour)
