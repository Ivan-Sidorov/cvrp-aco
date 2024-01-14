import glob
import itertools
import operator
import os
import pickle
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import vrplib
from tqdm import tqdm


def make_graph(weights: np.ndarray, coords: np.ndarray = None):
    vertices = np.arange(0, weights.shape[0])
    is_coords = True
    if not coords:
        is_coords = False
        coords = np.arange(weights.shape[0])  # делаем заглушку
    graph = dict(zip(vertices, coords))

    edges = {}
    pheromones = {}
    for idx, coords in graph.items():
        for idx_other, coords_other in graph.items():
            vertex_pair = (min(idx, idx_other), max(idx, idx_other))
            # считаем расстояния между всеми парами точек
            if is_coords:
                edge = np.sqrt(
                    (coords[0] - coords_other[0]) ** 2
                    + (coords[1] - coords_other[1]) ** 2
                )
            else:
                edge = weights[idx, idx_other]
            edges[vertex_pair] = edge

            # инициализируем феромоны единицей
            if idx != idx_other:
                pheromones[vertex_pair] = 0.1

    vertices = vertices[1:]  # убираем депо

    return vertices, edges, pheromones


def single_ant_solution(
    vertices, edges, init_capacity, demand, pheromones, alpha, beta
):
    vertices = vertices.copy()
    final_path = []
    curr_capacity = None

    while len(vertices) != 0:
        path = []
        vertex = np.random.choice(vertices)  # муравей начинает из случайной вершины
        curr_capacity = init_capacity - demand[vertex]  # оставшаяся вместимость
        path.append(vertex)
        vertices = vertices[vertices != vertex]

        while len(vertices) != 0:
            # числитель в формуле
            vfunc = np.vectorize(
                lambda x: ((pheromones[(min(x, vertex), max(x, vertex))]) ** alpha)
                * ((1 / edges[(min(x, vertex), max(x, vertex))]) ** beta)
            )
            probabilities = vfunc(vertices)
            probabilities = np.nan_to_num(probabilities, nan=0)

            # вероятности перехода муравья из текущей вершины в другие
            probabilities = probabilities / np.sum(probabilities)

            # переходим в случайную вершину с учетом вероятностей
            vertex = np.random.choice(vertices, p=probabilities)
            curr_capacity = curr_capacity - demand[vertex]  # оставшаяся вместимость

            # проверяем, что можем забрать посылку
            if curr_capacity > 0:
                path.append(vertex)
                vertices = vertices[vertices != vertex]
            else:
                break

        final_path.append(path)

    return final_path


def calc_distance(path, edges):
    final_distance = 0
    for single_path in path:
        start = 0  # стартуем из депо
        for vertex in single_path:
            end = vertex
            final_distance += edges[min(start, end), max(start, end)]
            start = end
        end = 0  # возвращаемся в депо
        final_distance += edges[min(start, end), max(start, end)]
    return final_distance


def update_pheromones(pheromones, solutions, distances, Q, rho):
    # обновление значений феромонов
    L = sum(distances) / len(solutions)  # среднее пройденное расстояние
    delta_pheromones = Q / L  # добавка феромона
    pheromones = {
        idx: (1 - rho) * val + delta_pheromones for (idx, val) in pheromones.items()
    }

    return pheromones


def update_best_path(paths, distances, best_path, best_distance):
    paths, distances = zip(*sorted(zip(paths, distances), key=lambda x: x[1]))
    if best_path != None:
        if distances[0] < best_distance:
            best_path = paths[0]
            best_distance = distances[0]
    else:
        best_path = paths[0]
        best_distance = distances[0]

    return best_path, best_distance


def optimize(instance, iterations, ants, alpha, beta, rho, Q):
    best_path = None
    best_distance = None
    vertices, edges, pheromones = make_graph(
        instance["edge_weight"]
    )  # instance["node_coord"]

    for i in range(iterations):
        paths = []
        distances = []
        for _ in range(ants):
            path = single_ant_solution(
                vertices,
                edges,
                instance["capacity"],
                instance["demand"],
                pheromones,
                alpha,
                beta,
            )
            paths.append(path)
            distances.append(calc_distance(path, edges))

        update_pheromones(pheromones, paths, distances, Q, rho)
        best_path, best_distance = update_best_path(
            paths, distances, best_path, best_distance
        )

    return best_path, best_distance


def calc_error(best_distance, optim_distance):
    return 100 * abs(optim_distance - best_distance) / optim_distance


def optimization(instance, solution, iterations, ants, comb):
    _, best_distance = optimize(
        instance,
        iterations=iterations,
        ants=ants,
        alpha=comb[0],
        beta=comb[1],
        rho=comb[2],
        Q=comb[3],
    )
    error = calc_error(best_distance, solution["cost"])
    return (comb, error)


def grid_search(instance, solution, iterations, ants, param_grid):
    params = list(param_grid.values())
    combinations = list(itertools.product(*params))
    with Pool(processes=7) as pool:
        collection = list(
            tqdm(
                pool.imap(
                    partial(optimization, instance, solution, iterations, ants),
                    combinations,
                ),
                total=len(combinations),
                desc="Grid-Search",
            )
        )

    collection.sort(key=operator.itemgetter(1))
    best_params = collection[0][0]

    return best_params


def make_grid_search(param_grid, iterations, ants, task_path):
    cnt = 0
    for path in glob.glob(task_path):
        cnt += 1

    all_params = {}
    for path in tqdm(glob.glob(task_path), total=cnt):
        try:
            path = Path(path)
            instance = vrplib.read_instance(path)
            solution = vrplib.read_solution(
                os.path.join(path.parent, path.stem + ".sol")
            )

            best_params = grid_search(instance, solution, iterations, ants, param_grid)
            all_params[str(path)] = best_params
        except ValueError:
            print(f"Optimization for {instance['name']} failed")

    return all_params


if __name__ == "__main__":
    iterations = 800
    ants = 20
    param_grid = {"alpha": [3, 7], "beta": [7, 12], "rho": [0.01, 0.2], "Q": [50, 200]}
    np.random.seed(1)
    all_params = make_grid_search(
        param_grid=param_grid,
        iterations=iterations,
        ants=ants,
        task_path="/Users/ivsidorov/Documents/Учеба/Семинар наставника/hw2/data/E/*.vrp",
    )
    with open("all_params_e.pkl", "wb") as f:
        pickle.dump(all_params, f)
