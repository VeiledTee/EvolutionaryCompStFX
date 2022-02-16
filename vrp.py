import numpy as np
from numpy import ndarray
import random
from typing import Any, List, Tuple
from copy import deepcopy
from itertools import cycle
import matplotlib.pyplot as plt
import math

MAX_GENERATIONS = 100
SELECTION_SIZE: int = 2
NUM_VEHICLES: int = 4
CROSSOVER_RATE: float = 0.8
MUTATION_RATE: float = 0.1

DATA: List[List[int]] = []
POPULATION: List[List[int]] = []
VRP_POPULATION: List[List[List[int]]] = []
POINTS: List[List[float]] = []
TSP_FITNESS: List[float] = []
VRP_FITNESS: List[float] = []
# FITNESS: List[float] = []


def vrp_initialize(to_encode: Any) -> None:
	global DATA, POINTS, VRP_POPULATION, VRP_FITNESS
	tsp_initialize(to_encode)

	vehicle_perm: List[List[int]] = [[] for _ in range(NUM_VEHICLES)]
	for k in range(NUM_VEHICLES):
		to_append: List = []
		for i in range(len(POPULATION)):
			if i % NUM_VEHICLES == k:
				to_append.append(POPULATION[i])
		vehicle_perm[k].extend(list(to_append))

	depot: int = vehicle_perm[0][0][0]
	for i in range(0, len(vehicle_perm)):
		if vehicle_perm[i][0][0] != depot:
			for j, k in enumerate(vehicle_perm[i][0]):
				if k == depot:
					vehicle_perm[i][0][0], vehicle_perm[i][0][j] = vehicle_perm[i][0][j], vehicle_perm[i][0][0]
	VRP_POPULATION = vehicle_perm
	print(f"VRP POP: {len(VRP_POPULATION)}")
	print(f"VRP POP[0]: {len(VRP_POPULATION[0])}")
	print(f"VRP POP[0][0]: {len(VRP_POPULATION[0][0])}")
	fitness = []
	for v in range(len(VRP_POPULATION)):
		fitness.append([])
		for i in range(len(VRP_POPULATION[v])):
			fitness[v].append(np.inf)
	VRP_FITNESS = fitness
	print(f"VRP Fit: {VRP_FITNESS}")

def tsp_initialize(to_encode: Any) -> None:
	"""
	Encodes either a .txt file or a list of lists of ints as the population
	:param to_encode: a .txt filename as a string, or an adjacency matrix represented as a list of lists of ints
	"""
	global DATA, POINTS, POPULATION
	if type(to_encode) == str:
		tsp_file = open(to_encode, 'r')
		lines = tsp_file.readlines()
		for line in lines:
			line = line.split()
			DATA.append([int(l) for l in line])
		for i in range(len(DATA)):
			POPULATION.append(np.random.permutation(len(DATA)).tolist())
			TSP_FITNESS.append(np.inf)
		tsp_file.close()
		temp_points = calculate_positions(np.asarray(DATA))
		for i in range(len(temp_points)):
			POINTS.append(list(temp_points[i]))
	elif type(to_encode) == list and type(to_encode[0]) == list:
		DATA = to_encode
	else:
		raise TypeError(f"The parameter passed is not of type str or List[List[int]]. Please reformat.")

def vrp_fitness_population(fit_func) -> None:
	for v in range(len(VRP_POPULATION)):
		for i in range(len(VRP_POPULATION[v])):
			VRP_FITNESS[v][i] = fit_func(VRP_POPULATION[v][i])

def vrp_fitness_manhattan(chromosome: List[List[int]]) -> float:
	# manhattan distance
	m_dist: float = 0.0
	for i in range(len(chromosome)):
		if i + 1 == len(chromosome):
			m_dist += abs(POINTS[chromosome[i]][0][chromosome[0]][0]) + abs(POINTS[chromosome[i]][1][chromosome[0]][1])
		else:
			m_dist += abs(POINTS[chromosome[i]][chromosome[0]]) + abs(POINTS[chromosome[i]][chromosome[0]])
	return m_dist

def vrp_fitness_euclidean(chromosome: List[List[int]]) -> float:
	# euclidean distance
	e_dist: float = 0.0
	for i in range(len(chromosome)):
		if i + 1 == len(chromosome):
			e_dist += math.sqrt((POINTS[chromosome[0]][0] - POINTS[chromosome[i]][0])**2 + (POINTS[chromosome[0]][1] - POINTS[chromosome[i]][1])**2)
		else:
			e_dist += math.sqrt((POINTS[chromosome[i+1]][0] - POINTS[chromosome[i]][0])**2 + (POINTS[chromosome[i+1]][1] - POINTS[chromosome[i]][1])**2)
	return e_dist

def tsp_fitness_population(fit_func) -> None:
	"""
	Calculate the fitness of an entire population
	:param fit_func: the fitness function to calculate the fitness of the chromosomes
	"""
	for index in range(len(POPULATION)):
		TSP_FITNESS[index] = fit_func(POPULATION[index])

def tsp_fitness_chromosome(chromosome: List[int]) -> int:
	"""
	Calculate the fitness of an individual chromosome
	:param chromosome: The chromosome to calculate the fitness for
	:return: fitness value
	"""
	fitness = 0
	for gene_dex in range(len(chromosome)):
		if gene_dex + 1 == len(chromosome):
			fitness += DATA[chromosome[gene_dex]][chromosome[-1]]
		else:
			fitness += DATA[chromosome[gene_dex]][chromosome[gene_dex + 1]]
	return fitness

def tsp_selection() -> int:
	"""
	Randomly select a SELECTION_SIZE number of chromosomes and find the best one
	:return: the index of the best of the selection
	"""
	best_index = np.inf
	best_fitness = np.inf
	for _ in range(SELECTION_SIZE):
		maybe_parent: int = random.randint(0, len(POPULATION) - 1)
		if TSP_FITNESS[maybe_parent] < best_fitness:
			best_index = maybe_parent
			best_fitness = TSP_FITNESS[maybe_parent]
	return best_index

def vrp_selection(vehicle_number: int) -> int:
	"""
	Randomly select a SELECTION_SIZE number of chromosomes and find the best one
	:return: the index of the best of the selection
	"""
	best_index = np.inf
	best_fitness = np.inf
	for v in range(NUM_VEHICLES):
		for _ in range(SELECTION_SIZE):
			maybe_parent: int = random.randint(0, len(VRP_POPULATION) - 1)
			if VRP_FITNESS[v][maybe_parent] < best_fitness:
				best_index = maybe_parent
				best_fitness = VRP_FITNESS[v][maybe_parent]
	return best_index

def mutate(function, chromosome: List[int]) -> None:
	function(chromosome)

def inversion(chromosome: List[int]):
	city_1, city_2 = random.randint(0, len(chromosome) - 1), random.randint(0, len(chromosome) - 1)
	while city_1 == city_2:  # same random int check
		city_2 = random.randint(0, len(chromosome) - 1)
		if city_1 != city_2:
			break
	invert: List[int] = chromosome[city_1:city_2]
	invert.reverse()
	for index, number in enumerate(invert):
		chromosome[city_1 + index] = number

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

def tsp_generation(filename: str, l_chromos: List[List[int]], vehicle: int):
	global POPULATION, TSP_FITNESS
	if filename[-4:] != '.txt':
		filename += '.txt'
	gen_min_values: List[int] = []
	gen_avg_values: List[float] = []

	for i in range(MAX_GENERATIONS):
		# if i % (MAX_GENERATIONS * .1) == 0:
		# 	print(f"TSP GENERATION {i}")

		tsp_fitness_population(tsp_fitness_chromosome)
		new_pop: List[List[int]] = []

		gen_min_values.append(np.min(TSP_FITNESS))
		gen_avg_values.append(np.average(TSP_FITNESS))

		for j in range(0, len(l_chromos), 2):
			index_1: int = tsp_selection()
			index_2: int = tsp_selection()
			while index_1 == index_2:
				index_2 = tsp_selection()
		# 	if j == len(POPULATION) - 1:
		# 		parent_1: List[int] = deepcopy(POPULATION[index_1])
		# 		if CROSSOVER_RATE > random.random():
		# 			crossover(parent_1, parent_1)
		# 		if MUTATION_RATE > random.random():
		# 			mutate(parent_1)
		# 		new_pop.append(parent_1)
		#
			parent_1: List[int] = deepcopy(VRP_POPULATION[index_1])
			parent_2: List[int] = deepcopy(VRP_POPULATION[index_2])

			if CROSSOVER_RATE > random.random():
				crossover(parent_1, parent_2)
			if MUTATION_RATE > random.random():
				mutate(inversion, parent_1)
				# mutate(random_swap, parent_1)
			if MUTATION_RATE > random.random():
				mutate(inversion, parent_2)
				# mutate(random_swap, parent_2)
			new_pop.append(parent_1)
			new_pop.append(parent_2)

		return new_pop

# tsp_fitness_population(tsp_fitness_chromosome)
#
# best_result: int = np.min(FITNESS)
# best_index: int = np.argmin(FITNESS)
# best_sln: List[int] = POPULATION[best_index - 1]
#
# return best_result, best_sln

def vrp_generation(filename: str):
	global POPULATION, TSP_FITNESS, VRP_POPULATION, VRP_FITNESS
	gen_min_values: List[float] = []
	gen_avg_values: List[float] = []
	vrp_initialize(filename)
	for i in range(MAX_GENERATIONS):
		if i % (MAX_GENERATIONS * .1) == 0:
			print(f"VRP GEN {i}")
			for f in VRP_FITNESS:
				print(f"FITNESS: {f}")

		vrp_fitness_population(vrp_fitness_euclidean)
		# vrp_fitness_population(vrp_fitness_manhattan)

		gen_min_values.append(np.min(VRP_FITNESS))
		gen_avg_values.append(np.average(VRP_FITNESS))

		for vehicle in range(NUM_VEHICLES):
			new_pop: List[List[List[int]]] = []
			for j in range(0, len(VRP_POPULATION[vehicle]), 2):
				if j == len(VRP_POPULATION[vehicle]) - 1:
					index_1: int = vrp_selection(vehicle)
					parent_1: List[int] = deepcopy(VRP_POPULATION[vehicle][index_1])
					if MUTATION_RATE > random.random():
						mutate(inversion, parent_1)
					new_pop.append(parent_1)
				else:
					index_1: int = vrp_selection(vehicle)
					index_2: int = vrp_selection(vehicle)
					while index_1 == index_2:
						index_2 = vrp_selection(vehicle)

					parent_1: List[int] = deepcopy(VRP_POPULATION[vehicle][index_1])
					parent_2: List[int] = deepcopy(VRP_POPULATION[vehicle][index_2])

					if CROSSOVER_RATE > random.random():
						crossover(parent_1, parent_2)
					if MUTATION_RATE > random.random():
						mutate(inversion, parent_1)
					# mutate(random_swap, parent_1)
					if MUTATION_RATE > random.random():
						mutate(inversion, parent_2)
					# mutate(random_swap, parent_2)
					new_pop.append(parent_1)
					new_pop.append(parent_2)

			VRP_POPULATION[vehicle] = new_pop
	vrp_visualization()

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
			P[j, 1] = - P[j, 1]
	return P

def vrp_visualization() -> None:
	f_sum: List[float] = []
	for i in range(len(VRP_FITNESS[NUM_VEHICLES-1])):
		vehicle_sum: float = 0
		for v in range(len(VRP_FITNESS)):
			vehicle_sum += VRP_FITNESS[v][i]
		f_sum.append(vehicle_sum)
	best_dex: int = np.argmin(f_sum)
	print(f"best dex: {best_dex}")
	best_sln = []
	for v in range(len(VRP_FITNESS)):
		print(f"V Val: {v}")
		best_sln.append(VRP_POPULATION[v][best_dex])
	print(len(best_sln))
	plt.show()


if __name__ == '__main__':
	np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
	# partially mapped vs ordered
	# results: List[int] = []
	# solutions: List[List[int]] = []
	# x, y = tsp_generation('tsp_test_2.txt', f'tsp_test_1_image')
	# results.append(x)
	# solutions.append(y)
	# reset_globals()
	# print(results)
	# print(min(results))
	vrp_generation('tsp_test_2.txt')
	print(len(VRP_FITNESS))
	print(len(VRP_FITNESS[0]))
	# print(len(VRP_FITNESS[0][0]))
	print(len(VRP_POPULATION))
	print(len(VRP_POPULATION[0]))
	print(len(VRP_POPULATION[0][0]))
	"""
initialize/get data/etc
for generation in MAX_GENERATIONS:
	new_population = []
	find fitness
	append min/avg values
	for every second thing in the length of the population:
		select 2 different parents
		crossover
		mutation
	population = new_population
"""