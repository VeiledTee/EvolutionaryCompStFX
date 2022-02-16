import math
import random
from copy import deepcopy
from itertools import cycle
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import numpy as np
from numpy import ndarray

MAX_GENERATIONS = 100
SELECTION_SIZE: int = 20
POPULATION_SIZE: int = 200
CROSSOVER_RATE: float = 0.8
MUTATION_RATE: float = 0.2
ELITISM: int = 20

DATA: List[List[int]] = []
POPULATION: List[List[int]] = []
FITNESS: List[float] = []
POINTS: List[List[float]] = []


def euclidean_dist(p1: List[float], p2: List[float]) -> float:
	return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def initialize_cords(to_encode: Any) -> None:
	"""
	Encodes either a .txt file or a list of lists of ints as the population
	:param to_encode: a .txt filename as a string, or an adjacency matrix represented as a list of lists of ints
	"""
	global DATA
	if type(to_encode) == str:
		tsp_file = open(to_encode, "r")
		lines = tsp_file.readlines()
		for line in lines:
			line = line.strip().split()
			POINTS.append([float(line[1]), float(line[2])])
		for _ in range(POPULATION_SIZE):
			POPULATION.append(np.random.permutation(len(POINTS)).tolist())
			FITNESS.append(np.inf)
		tsp_file.close()
		DATA = [[] for _ in range(len(POINTS))]
		for i in range(len(POINTS)):
			for j in range(len(POINTS)):
				if i == j:
					DATA[i].append(0)
				else:
					DATA[i].append(euclidean_dist(POINTS[i], POINTS[j]))
	elif type(to_encode) == list and type(to_encode[0]) == list:
		DATA = to_encode
	else:
		raise TypeError(f"The parameter passed is not of type str or List[List[int]]. Please reformat.")


def initialize(to_encode: Any) -> None:
	"""
	Encodes either a .txt file or a list of lists of ints as the population
	:param to_encode: a .txt filename as a string, or an adjacency matrix represented as a list of lists of ints
	"""
	global DATA
	if type(to_encode) == str:
		tsp_file = open(to_encode, "r")
		lines = tsp_file.readlines()
		for line in lines:
			line = line.split()
			DATA.append([int(l) for l in line])
		for _ in range(POPULATION_SIZE):
			POPULATION.append(np.random.permutation(len(DATA)).tolist())
			FITNESS.append(np.inf)
		tsp_file.close()
		temp_points = calculate_positions(np.asarray(DATA))
		for i in range(len(temp_points)):
			POINTS.append(list(temp_points[i]))
	elif type(to_encode) == list and type(to_encode[0]) == list:
		DATA = to_encode
	else:
		raise TypeError(f"The parameter passed is not of type str or List[List[int]]. Please reformat.")


def fitness_population(fit_func) -> None:
	"""
	Calculate the fitness of an entire population
	:param fit_func: the fitness function to calculate the fitness of the chromosomes
	"""
	for index in range(len(POPULATION)):
		if index >= len(FITNESS):
			pass
		else:
			FITNESS[index] = fit_func(POPULATION[index])


def fitness_chromosome(chromosome: List[int]) -> int:
	"""
	Calculate the fitness of an individual chromosome
	:param chromosome: The chromosome to calculate the fitness for
	:return: fitness value
	"""
	fitness = 0
	for gene_dex, gene in enumerate(chromosome):
		if gene_dex >= len(chromosome) - 1:
			fitness += DATA[chromosome[gene_dex]][chromosome[0]]
		else:
			fitness += DATA[chromosome[gene_dex]][chromosome[gene_dex + 1]]
	return fitness


def selection() -> int:
	"""
	Randomly select a SELECTION_SIZE number of chromosomes and find the best one
	:return: the index of the best of the selection
	"""
	best_index = np.inf
	best_fitness = np.inf
	for _ in range(SELECTION_SIZE):
		maybe_parent: int = random.randint(0, POPULATION_SIZE - ELITISM)
		if FITNESS[maybe_parent] < best_fitness:
			best_index = maybe_parent
			best_fitness = FITNESS[maybe_parent]
	return best_index


def mutate(function, chromosome: List[int]) -> None:
	function(chromosome)


def random_swap(chromosome: List[int]) -> None:
	"""
	Mutate a chromosome by swapping two cities in the permutation
	:param chromosome: chromosome to mutate
	"""
	city_1, city_2 = random.randint(0, len(chromosome) - 1), random.randint(0, len(chromosome) - 1)
	while city_1 == city_2:  # same random int check
		city_2 = random.randint(0, len(chromosome) - 1)
		if city_1 != city_2:
			break
	chromosome[city_1], chromosome[city_2] = chromosome[city_2], chromosome[city_1]  # fancy python swap


def inversion(chromosome: List[int]):
	"""
	Invert a section of a chromosome
	"""
	city_1, city_2 = random.randint(0, len(chromosome) - 1), random.randint(0, len(chromosome) - 1)
	while city_1 == city_2:  # same random int check
		city_2 = random.randint(0, len(chromosome) - 1)
		if city_1 != city_2:
			break
	invert: List[int] = chromosome[city_1:city_2]
	invert.reverse()
	for index, number in enumerate(invert):
		chromosome[city_1 + index] = number


def preserve_permutation(index: int, permutation_1: List[int], permutation_2: List[int]) -> None:
	"""
	Preserve the uniqueness of a chromosome during one-point crossover
	:param index: index to crossover
	:return: 2 crossed chromosomes
	"""
	start_1: List[int] = permutation_1[:index]
	start_2: List[int] = permutation_2[:index]
	to_return_1: List[int] = deepcopy(start_1)
	to_return_2: List[int] = deepcopy(start_2)
	for permutation, to_ret in ((permutation_1, to_return_2), (permutation_2, to_return_1)):
		passed = 0
		cycling = cycle(permutation)
		for _ in range(index):
			next(cycling)
		for value in cycling:
			if value in to_ret:
				passed += 1
				pass
			else:
				to_ret.append(value)
			if passed == len(permutation):
				break
	permutation_1 = to_return_1
	permutation_2 = to_return_2


def crossover(chromosome_1: List[int], chromosome_2: List[int]) -> None:
	"""
	Choose a random index and perform one point crossover between the two chromosome
	:param chromosome_1: first of the two chromosomes to crossover
	:param chromosome_2: second of the two chromosomes to crossover
	"""
	index: int = random.randint(0, len(chromosome_1))  # randomly select index
	preserve_permutation(index, chromosome_1, chromosome_2)


def reset_globals():
	global DATA, POPULATION, FITNESS, POINTS
	DATA = []
	POPULATION = []
	FITNESS = []
	POINTS = []


def x_coord_of_point(D, j):
	return (D[0, j] ** 2 + D[0, 1] ** 2 - D[1, j] ** 2) / (2 * D[0, 1])


def coords_of_point(D, j) -> ndarray:
	x = x_coord_of_point(D, j)
	return np.array([x, math.sqrt(abs(D[0, j] ** 2 - x ** 2))], dtype=object)


def calculate_positions(D) -> ndarray:
	(m, n) = D.shape
	P = np.zeros((n, 2))
	tr = (min(min(D[2, 0:2]), min(D[2, 3:n])) / 2) ** 2
	P[1, 0] = D[0, 1]
	P[2, :] = coords_of_point(D, 2)
	for j in range(3, n):
		P[j, :] = coords_of_point(D, j)
		if abs(np.dot(P[j, :] - P[2, :], P[j, :] - P[2, :]) - D[2, j] ** 2) > tr:
			P[j, 1] = -P[j, 1]
	return P


def visualize(filename: str = ""):
	best_index: int = np.argmin(FITNESS)
	best_sln: List[int] = POPULATION[best_index]
	x = [POINTS[a][0] for a in best_sln]
	x.append(x[0])
	y = [POINTS[b][1] for b in best_sln]
	y.append(y[0])
	fig, ax = plt.subplots()
	(_,) = ax.plot(x, y, "-o")
	if filename == "":
		fig.savefig(f"Final_Graph{filename}.svg", format="svg")
	else:
		fig.savefig(f"Final_Graph_{filename}.svg", format="svg")
	plt.show()
	plt.close(fig)


def stats_comparison(list_1: list, list_2: list) -> Tuple[float, float]:
	stat, pval = ttest_ind(list_1, list_2)
	return stat, pval


def genetics(filename: str, save_file: str = "", elite: bool = True) -> Tuple[int, List[int]]:
	global POPULATION, FITNESS
	if filename[-4:] != ".txt":
		initialize_cords(filename)
	else:
		initialize(filename)

	gen_min_values: List[float] = []
	gen_avg_values: List[float] = []

	for i in range(MAX_GENERATIONS):
		if i % (MAX_GENERATIONS * 0.1) == 0:
			print(f"GENERATION {i}")

		fitness_population(fitness_chromosome)
		new_pop: List[List[int]] = []

		gen_min_values.append(min(FITNESS))
		gen_avg_values.append(np.average(FITNESS))
		# Elitism - guarantee top x in next gen (usually 1)
		if elite:
			for _ in range(ELITISM):
				new_pop.append(POPULATION[np.argmin(FITNESS)])
		for j in range(0, len(POPULATION), 2):
			index_1: int = selection()
			index_2: int = selection()
			while index_1 == index_2:
				index_2 = selection()
			if j == len(POPULATION) - 1:
				parent_1: List[int] = deepcopy(POPULATION[index_1])
				if CROSSOVER_RATE > random.random():
					crossover(parent_1, parent_1)
				if MUTATION_RATE > random.random():
					if random.random() > 0.5:
						mutate(random_swap, parent_1)
					else:
						mutate(inversion, parent_1)
				new_pop.append(parent_1)

			parent_1: List[int] = deepcopy(POPULATION[index_1])
			parent_2: List[int] = deepcopy(POPULATION[index_2])

			if CROSSOVER_RATE > random.random():
				crossover(parent_1, parent_2)
			if MUTATION_RATE > random.random():
				# mutate(inversion, parent_1)
				mutate(random_swap, parent_1)
			if MUTATION_RATE > random.random():
				# mutate(inversion, parent_2)
				mutate(random_swap, parent_2)
			new_pop.append(parent_1)
			new_pop.append(parent_2)

		POPULATION = new_pop

	fitness_population(fitness_chromosome)

	best_result: int = np.min(FITNESS)
	best_index: int = np.argmin(FITNESS)
	best_sln: List[int] = POPULATION[best_index - 1]

	# fig, ax = plt.subplots()
	# ax.plot(gen_min_values)
	# ax.plot(gen_avg_values)
	# fig.legend(["Min Values", "Avg Values"])
	# fig.savefig(f"GA_Min_Avg_Trend_{save_file}.svg", format="svg")
	# plt.show()
	# plt.close(fig)

	return best_result, best_sln


if __name__ == "__main__":
	x, y = genetics(filename="compete.cords")
	print(x)
	print(y)
	# visualize("Final_Graph")
	# _, _ = genetics(filename="example.cords", elite=False)
	# visualize("Final_Graph")
	# initialize_cords('example.cords')
	# print(FITNESS)
