import numpy as np
import pandas as pd
import random
from typing import Tuple, Any
import matplotlib.pyplot as plt

# Parameters - change these!
NUM_PARTICLES: int = 30
C1: float = 0.1
C2: float = 0.1
W: float = 0.8
EXCTINCTION_CHANCE: float = .25
EXCTINCTION_PERCENTAGE: float = 0.75
VELOCITY_MAX: float = 0.25
VELOCITY_CLAMP: bool = False

# Parameters - plez don't touch!!
PARTICLES = np.random.randn(2, NUM_PARTICLES) * 5
VELOCITIES = np.random.randn(2, NUM_PARTICLES) * 0.1
PBEST: np.ndarray = PARTICLES
PBEST_FITNESS: np.ndarray = np.ndarray([])
GBEST: np.ndarray = np.ndarray([])
GBEST_FITNESS: np.ndarray = np.ndarray([])

def ackley(x: Any, y: Any) -> Any:
	return -20.0 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20

def extinction_event(evaluation, minimum: float, maximum: float) -> None:
	global PARTICLES
	for i in range(int(30 * EXCTINCTION_PERCENTAGE)):
		PARTICLES[0][PARTICLES.shape[1] - (1 + i)] = float(np.random.uniform(minimum, maximum))
		PARTICLES[1][PARTICLES.shape[1] - (1 + i)] = float(np.random.uniform(minimum, maximum))
	update_velocity(evaluation)

def island_setup(num_islands: int) -> None:
	global PARTICLES
	while True:
		if NUM_PARTICLES % num_islands == 0:
				PARTICLES = PARTICLES.reshape(num_islands, NUM_PARTICLES/num_islands)
				break
		else:
			num_islands = input(f'Please input a multiple of {NUM_PARTICLES}: ')
			num_islands = int(num_islands)

def update_particle(function, particle_num: int) -> None:
	if function(PARTICLES[particle_num][0][0], PARTICLES[particle_num][0][1]) > PARTICLES[particle_num][2]:
		PARTICLES[particle_num][2] = function(PARTICLES[particle_num][0][0], PARTICLES[particle_num][0][1])

def pso_initialize(evaluation) -> None:
	global PBEST, GBEST, PBEST_FITNESS, GBEST_FITNESS
	PBEST_FITNESS = evaluation(PARTICLES[0], PARTICLES[1])
	GBEST = PBEST[:, PBEST_FITNESS.argmin()]
	GBEST_FITNESS = PBEST_FITNESS.min()

def update_velocity(eval_function) -> np.ndarray:
	"""
	Function to do one iteration of particle swarm optimization
	"""
	global VELOCITIES, PARTICLES, PBEST, GBEST
	# Update params
	r1, r2 = np.random.rand(2)
	VELOCITIES = (W * VELOCITIES) + (C1 * r1 * (PBEST - PARTICLES)) + (C2 * r2 * (GBEST.reshape(-1, 1) - PARTICLES))
	PARTICLES = PARTICLES + VELOCITIES
	return eval_function(PARTICLES[0], PARTICLES[1])

def position_clamping(max_value: Any) -> int:
	return 0

def velocity_clamping(max_value: Any) -> None:
	VELOCITIES[VELOCITIES > max_value] = max_value

def update_velocity(eval_function, clamp: bool = False) -> np.ndarray:
	"""
	Function to do one iteration of particle swarm optimization
	"""
	global VELOCITIES, PARTICLES, PBEST, GBEST
	# Update params
	r1, r2 = np.random.rand(2)
	VELOCITIES = (W * VELOCITIES) + (C1 * r1 * (PBEST - PARTICLES)) + (C2 * r2 * (GBEST.reshape(-1, 1) - PARTICLES))
	if clamp:
		velocity_clamping(VELOCITY_MAX)
	PARTICLES = PARTICLES + VELOCITIES
	return eval_function(PARTICLES[0], PARTICLES[1])

def reset_globals() -> None:
	global PARTICLES, VELOCITIES, PBEST, PBEST_FITNESS, GBEST, GBEST_FITNESS
	PARTICLES = np.random.randn(2, NUM_PARTICLES) * 5
	VELOCITIES = np.random.randn(2, NUM_PARTICLES) * 0.1
	PBEST = PARTICLES
	PBEST_FITNESS = np.ndarray([])
	GBEST = np.ndarray([])
	GBEST_FITNESS = np.ndarray([])

def factors(x: int) -> Tuple[int, int]:
	"""
	Return the factors of x as a tuple
	:param x: The number to find the factors of
	:return: A tuple containing the two factors of x that are the closest in value
	"""
	factors_of = []
	for i in range(1, x + 1):
		if x % i == 0:
			factors_of.append(i)
	if len(factors_of) % 2 == 0:
		return factors_of[(len(factors_of)//2) - 1], factors_of[len(factors_of)//2]
	else:
		return factors_of[len(factors_of)//2], factors_of[len(factors_of)//2]

def bounds(evaluation) -> Tuple[float, float, float, float]:
	# Plot in 3D and find the global minimum
	x, y = np.array(np.meshgrid(np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000)))
	z = evaluation(x, y)
	x_min = x.ravel()[z.argmin()]
	y_min = y.ravel()[z.argmin()]
	x_max = x.ravel()[z.argmax()]
	y_max = y.ravel()[z.argmax()]
	plt.figure(figsize=(8, 6))
	plt.imshow(z, extent=[-25, 25, -25, 25], origin='lower', cmap='hot', alpha=0.5)
	plt.colorbar()
	plt.title(f"Colour map of the {evaluation.__name__} function")
	plt.plot([x_min], [y_min], marker='x', markersize=5, color="white")
	contours = plt.contour(x, y, z, 10, colors='black', alpha=0.4)
	plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
	plt.savefig(f'base_{evaluation.__name__}.svg', format='svg')
	plt.close()
	return x_min, y_min, x_max, y_max

def plot_cur_iteration(iteration: int, evaluation, filename: str) -> None:
	# Plot in 3D and find the global minimum
	x, y = np.array(np.meshgrid(np.linspace(PARTICLES.min(), PARTICLES.max(), 1000), np.linspace(PARTICLES.min(), PARTICLES.max(), 1000)))
	z = evaluation(x, y)
	x_min = x.ravel()[z.argmin()]
	y_min = y.ravel()[z.argmin()]
	fig, ax = plt.subplots(figsize=(8, 6))
	fig.set_tight_layout(True)
	img = ax.imshow(z, extent=[PARTICLES.min(), PARTICLES.max(), PARTICLES.min(), PARTICLES.max()], origin='lower', cmap='hot', alpha=0.5)
	fig.colorbar(img, ax=ax)
	ax.plot([x_min], [y_min], marker='x', markersize=5, color="white")
	contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
	ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
	pbest_plot = ax.scatter(PBEST[0], PBEST[1], marker='o', color='black', alpha=0.5)
	p_plot = ax.scatter(PARTICLES[0], PARTICLES[1], marker='o', color='blue', alpha=0.5)
	p_arrow = ax.quiver(PARTICLES[0], PARTICLES[1], VELOCITIES[0], VELOCITIES[1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
	gbest_plot = plt.scatter([GBEST[0]], [GBEST[1]], marker='*', s=100, color='black', alpha=0.4)
	plt.title(f"Iteration {iteration}")
	# plt.colorbar()
	plt.plot([x_min], [y_min], marker='x', markersize=5, color="white")
	contours = plt.contour(x, y, z, 10, colors='black', alpha=0.4)
	plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
	plt.savefig(f'base_{evaluation.__name__}_{filename}_{iteration}.svg', format='svg')
	plt.close()

def single_objective_pso(evaluation, filename: str, num_iter: int = 10000, vel_clamp: bool=False):
	global PARTICLES, PBEST, PBEST_FITNESS, GBEST, GBEST_FITNESS
	# initialize variables and set globals
	iteration: int = 0
	found_sln: bool = False
	pso_initialize(evaluation)
	min_x, min_y, max_x, max_y = bounds(evaluation)
	output: np.ndarray = np.zeros(num_iter)
	# evaluation loop
	while iteration < num_iter and not found_sln:
		output[iteration] = evaluation(*GBEST)
		if iteration == 0:
			plot_cur_iteration(iteration, evaluation, filename)
		# perform velocity calculation
		results = update_velocity(evaluation, vel_clamp)
		# update personal best with bool mask check
		PBEST[:, (PBEST_FITNESS >= results)] = PARTICLES[:, (PBEST_FITNESS >= results)]
		# update fitness values of particles
		PBEST_FITNESS = np.array([PBEST_FITNESS, results]).min(axis=0)
		# update global best
		GBEST = PBEST[:, PBEST_FITNESS.argmin()]
		# update global best fitness
		GBEST_FITNESS = PBEST_FITNESS.min()
		# extinction level event??
		if EXCTINCTION_CHANCE > np.random.uniform(0, 1):
			extinction_event(evaluation, min(min_x, min_y), max(max_x, max_y))
		# check to see if last 10 fitness values are the same
		if iteration > 10:
			prev_output = output[iteration - 10:iteration]
			if prev_output.any() == evaluation(*GBEST):
				plot_cur_iteration(iteration, evaluation, filename)
				break
		iteration += 1
	print(f"After {iteration} iterations...\nPSO found best solution at {evaluation.__name__}({GBEST[0]}, {GBEST[1]}) = {evaluation(*GBEST)}\n")

if __name__ == '__main__':
	single_objective_pso(ackley, filename='test', vel_clamp=VELOCITY_CLAMP)
