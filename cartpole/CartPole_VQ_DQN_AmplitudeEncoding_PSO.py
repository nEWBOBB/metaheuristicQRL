import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

import torch
import torch.nn as nn 
from torch.autograd import Variable
import itertools

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
from collections import deque

import gym
import time
import random
from collections import namedtuple
from copy import deepcopy

import os
import math
import copy

## PennyLane Part ##
# Specify the datatype of the Totch tensor
dtype = torch.DoubleTensor

## Define a FOUR qubit system
dev = qml.device('default.qubit', wires=2)
# dev = qml.device('qiskit.basicaer', wires=4)

def layer(W):
	""" Single layer of the variational classifier.

	Args:
		W (array[float]): 2-d array of variables for one layer
	"""

	qml.CNOT(wires=[0, 1])
	# qml.CNOT(wires=[1, 2])
	# qml.CNOT(wires=[2, 3])

	# qml.CNOT(wires=[3, 4])
	# qml.CNOT(wires=[4, 5])


	qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
	qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
	# qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
	# qml.Rot(W[3, 0], W[3, 1], W[3, 2], wires=3)

	# qml.Rot(W[4, 0], W[4, 1], W[4, 2], wires=4)
	# qml.Rot(W[5, 0], W[5, 1], W[5, 2], wires=5)


@qml.qnode(dev, interface='torch')
def circuit(weights, angles=None):
	"""The circuit of the variational classifier."""
	# Can consider different expectation value
	# PauliX , PauliY , PauliZ , Identity  

	qml.QubitStateVector(angles, wires=[0, 1])
	
	for W in weights:
		layer(W)
	
	return [qml.expval(qml.PauliZ(ind)) for ind in range(2)]
	


def variational_classifier(var_Q_circuit, var_Q_bias , angles=None):
	"""The variational classifier."""

	# Change to SoftMax???

	weights = var_Q_circuit
	# bias_1 = var_Q_bias[0]
	# bias_2 = var_Q_bias[1]
	# bias_3 = var_Q_bias[2]
	# bias_4 = var_Q_bias[3]
	# bias_5 = var_Q_bias[4]
	# bias_6 = var_Q_bias[5]

	# raw_output = circuit(weights, angles=angles) + np.array([bias_1,bias_2,bias_3,bias_4,bias_5,bias_6])

	angles = angles / np.sqrt(np.sum(angles ** 2))

	raw_output = circuit(weights, angles=angles) * var_Q_bias

	# raw_output = circuit(weights, angles=angles) + var_Q_bias
	# We are approximating Q Value
	# Maybe softmax is no need
	# softMaxOutPut = np.exp(raw_output) / np.exp(raw_output).sum()

	return raw_output

def return_random_agents(num_agents, num_qubits, num_layers):

	agent_list = []

	for ind in range(num_agents):
		np.random.seed(int(datetime.now().strftime("%S%f")))
		var_init_circuit = Variable(torch.tensor(0.01 * np.random.randn(num_layers, num_qubits, 3), device='cpu').type(dtype), requires_grad=False)
		var_init_bias = Variable(torch.tensor([1.0, 1.0], device='cpu').type(dtype), requires_grad=False)
		agent_list.append((var_init_circuit,var_init_bias))

	return agent_list

def run_agents(agents):

	reward_agents = []
	max_steps = 999999

	for agent_circuit_params in agents:
		var_Q_circuit = agent_circuit_params[0]
		var_Q_bias = agent_circuit_params[1]

		env = gym.make('CartPole-v1')
		n_actions = env.action_space.n
		# print("NUMBER OF ACTIONS:" + str(n_actions))

		s = env.reset()
		a = torch.argmax(variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles = s)).item()
		# a = epsilon_greedy(var_Q_circuit = agent_circuit_params[0], var_Q_bias = agent_circuit_params[1], epsilon = epsilon, n_actions = n_actions, s = s).item()
		t = 0
		total_reward = 0
		done = False

		while t < max_steps:
			t += 1
			# Execute the action 
			s_, reward, done, _ = env.step(a)

			# print("Reward : " + str(reward))
			# print("Done : " + str(done))
			total_reward += reward
			# a_ = np.argmax(Q[s_, :])
			a_ = torch.argmax(variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles = s_)).item()
			
			# print("ACTION:")
			# print(a_)

			s, a = s_, a_

			if done:
				# if render:
				# 	print("###FINAL RENDER###")
				# 	env.render()
				# 	print("###FINAL RENDER###")
				# 	print(f"This episode took {t} timesteps and reward: {total_reward}")
				# epsilon = epsilon / ((episode/10.) + 1)
				# print("Q Circuit Params:")
				# print(var_Q_circuit)
				# print(f"This episode took {t} timesteps and reward: {total_reward}")
				# timestep_reward.append(total_reward)
				# iter_index.append(episode)
				# iter_reward.append(total_reward)
				# iter_total_steps.append(t)
				# save_all_the_current_info(exp_name, file_title, episode, var_Q_circuit, var_Q_bias, iter_reward)
				break

		reward_agents.append(total_reward)

	return reward_agents

def return_average_score(agent, runs):
	score = 0.
	for i in range(runs):
		score += run_agents([agent])[0]
	return score/runs

def run_agents_n_times(agents, runs):
	avg_score = []
	for agent in agents:
		avg_score.append(return_average_score(agent,runs))
	return avg_score

def run_agent_n_times(agent, runs):
	avg_score = []
	avg_score.append(return_average_score(agent,runs))
	return avg_score

def mutate(agent):
	child_agent_var_Q_circuit = agent[0].clone().detach()
	child_agent_var_Q_bias = agent[1].clone().detach()

	
	mutation_power = 0.02 #hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf
			
	child_agent_var_Q_circuit += mutation_power * torch.randn_like(child_agent_var_Q_circuit)
	child_agent_var_Q_bias += mutation_power * torch.randn_like(child_agent_var_Q_bias)

	child_agent = (child_agent_var_Q_circuit, child_agent_var_Q_bias)

	return child_agent


def saveresults(iteration, num_particles, best_scores, raw_scores, raw_scores_selected, runtimes, inertia_weight, cognitive_weight, social_weight):
	#best_scores = [score[0] for score in best_scores]
	#raw_scores = [score[0] for score in raw_scores]
	#raw_scores_selected = [score[0] for score in raw_scores_selected]	

	iter_list = list(range(1, len(raw_scores) + 1))

	assert len(raw_scores) == len(raw_scores_selected) == len(best_scores)
	# Erstelle ein DataFrame mit den Daten
	data = {"raw": raw_scores, "rawselected": raw_scores_selected, "best": best_scores, "generations": iter_list, "runtime": runtimes}
	df = pd.DataFrame(data)

	# Speichere das DataFrame in einer CSV-Datei im Ordner "results" im selben Verzeichnis wie das Skript
	script_dir = os.path.dirname(os.path.abspath(__file__))

	# Erhalte den aktuellen Zeitpunkt und formatiere ihn als String
	now = datetime.now()
	timestamp = now.strftime('%m-%d_%H-%M-%S')

	# Erstelle den Ordner "results", wenn er noch nicht existiert
	results_dir = os.path.join(script_dir, 'results')
	results_dir = results_dir + '/CartPole_VQ_DQN_PSO'
	os.makedirs(results_dir, exist_ok=True)

	# Erstelle den Dateinamen mit dem Zeitstempel
	csv_filename = f'CartPole_VQ_DQN_PSO_NumParticles_{num_particles}_Inertia_{inertia_weight}_Cognitive_{cognitive_weight}_Social_{social_weight}_Iter_{iteration}_TIME{timestamp}.csv'
	csv_path = os.path.join(results_dir, csv_filename)

	df.to_csv(csv_path, index=False)
"""
def pso(agents, num_iterations, num_particles, inertia_weight, cognitive_weight, social_weight, raw_scores, raw_scores_selected, best_scores, runtimes, start_time):
	num_agents = len(agents)
	assert num_agents == num_particles, "Die Anzahl der Agenten muss mit der Anzahl der Partikel übereinstimmen."
   
	particle_positions = agents
	particle_velocities = [(torch.zeros_like(agent[0]), torch.zeros_like(agent[1])) for agent in agents]
	particle_best_positions = copy.deepcopy(agents)
	particle_best_scores = run_agents_n_times(agents, 3)
	global_best_position = copy.deepcopy(agents[np.argmax(particle_best_scores)])
	global_best_score = np.max(particle_best_scores)


	runtime = 0.0
	max_runtime = 36000.0 # 6 hours

	for iteration in range(num_iterations):	
		if runtime<max_runtime: 
			print("ITERATION: ", iteration)
		
			for i in range(num_particles):
				# Update particle velocity
				r1 = random.random()
				r2 = random.random()
			
				cognitive_term_circuit = cognitive_weight * r1 * (particle_best_positions[i][0] - particle_positions[i][0])
				social_term_circuit = social_weight * r2 * (global_best_position[0] - particle_positions[i][0])
				particle_velocities[i] = (inertia_weight * particle_velocities[i][0] + cognitive_term_circuit + social_term_circuit,
										inertia_weight * particle_velocities[i][1] + cognitive_weight * r1 * (particle_best_positions[i][1] - particle_positions[i][1]) + social_weight * r2 * (global_best_position[1] - particle_positions[i][1]))
			
				# Update particle position
				particle_positions[i] = (particle_positions[i][0] + particle_velocities[i][0],
										particle_positions[i][1] + particle_velocities[i][1])
			
				# Evaluate the new position
				new_score = run_agent_n_times(particle_positions[i], 3)
				print("New Score: ", new_score)
				# Update particle best position and score
				if new_score[0] > particle_best_scores[i]:
					particle_best_positions[i] = copy.deepcopy(particle_positions[i])
					particle_best_scores[i] = new_score[0]
			
				# Update global best position and score
				if new_score[0] > global_best_score:
					global_best_position = copy.deepcopy(particle_positions[i])
					global_best_score = new_score[0]
					print("New Best Score: ", global_best_score)

				print("Best Score: ", global_best_score)
			print("--------------------")
			raw_scores.append(new_score)
			raw_scores_selected.append(new_score)
			best_scores.append(global_best_score)
			runtimes.append(time.time() - start_time)


			print("Runtime: ", runtimes[-1])
			runtime = runtimes[-1]


			# Save the results every 50 iterations
			if iteration!=0 and iteration % 5000 == 0:
				saveresults(iteration, num_particles, best_scores, raw_scores, raw_scores_selected, runtimes, inertia_weight, cognitive_weight, social_weight)
		else:
			break	
	return global_best_position, global_best_score, raw_scores, raw_scores_selected, best_scores, runtimes
"""

def pso(agents, num_iterations, num_particles, inertia_weight, cognitive_weight, social_weight, raw_scores, raw_scores_selected, best_scores, runtimes, start_time):
	num_agents = len(agents)
	assert num_agents == num_particles, "Die Anzahl der Agenten muss mit der Anzahl der Partikel übereinstimmen."
   
	particle_positions = agents
	particle_velocities = [(torch.zeros_like(agent[0]), torch.zeros_like(agent[1])) for agent in agents]
	particle_best_positions = copy.deepcopy(agents)
	particle_best_scores = run_agents_n_times(agents, 3)
	global_best_position = copy.deepcopy(agents[np.argmax(particle_best_scores)])
	global_best_score = np.max(particle_best_scores)

	runtime = 0.0
	max_runtime = 20000.0 # ca 5 hours
	iteration = 0

	while runtime < max_runtime: 
		print("ITERATION: ", iteration)
	
		for i in range(num_particles):
			# Update particle velocity
			r1 = random.random()
			r2 = random.random()
		
			cognitive_term_circuit = cognitive_weight * r1 * (particle_best_positions[i][0] - particle_positions[i][0])
			social_term_circuit = social_weight * r2 * (global_best_position[0] - particle_positions[i][0])
			particle_velocities[i] = (inertia_weight * particle_velocities[i][0] + cognitive_term_circuit + social_term_circuit,
									inertia_weight * particle_velocities[i][1] + cognitive_weight * r1 * (particle_best_positions[i][1] - particle_positions[i][1]) + social_weight * r2 * (global_best_position[1] - particle_positions[i][1]))
		
			# Update particle position
			particle_positions[i] = (particle_positions[i][0] + particle_velocities[i][0],
									particle_positions[i][1] + particle_velocities[i][1])
		
			# Evaluate the new position
			new_score = run_agent_n_times(particle_positions[i], 3)
			print("New Score: ", new_score)
			# Update particle best position and score
			if new_score[0] > particle_best_scores[i]:
				particle_best_positions[i] = copy.deepcopy(particle_positions[i])
				particle_best_scores[i] = new_score[0]
		
			# Update global best position and score
			if new_score[0] > global_best_score:
				global_best_position = copy.deepcopy(particle_positions[i])
				global_best_score = new_score[0]
				print("New Best Score: ", global_best_score)

			print("Best Score: ", global_best_score)
		print("--------------------")
		raw_scores.append(new_score)
		raw_scores_selected.append(new_score)
		best_scores.append(global_best_score)
		runtimes.append(time.time() - start_time)


		print("Runtime: ", runtimes[-1])
		runtime = runtimes[-1]


		# Save the results every 50 iterations
		if iteration!=0 and iteration % 5000 == 0:
			saveresults(iteration, num_particles, best_scores, raw_scores, raw_scores_selected, runtimes, inertia_weight, cognitive_weight, social_weight)
			
		iteration += 1
	saveresults(iteration, num_particles, best_scores, raw_scores, raw_scores_selected, runtimes, inertia_weight, cognitive_weight, social_weight)
	return global_best_position, global_best_score, raw_scores, raw_scores_selected, best_scores, runtimes

def subtract_state_dicts(dict1, dict2):
	result = {}
	for key in dict1.keys():
		result[key] = dict1[key] - dict2[key]
	return result

def add_state_dicts(dict1, dict2):
	result = {}
	for key in dict1.keys():
		result[key] = dict1[key] + dict2[key]
	return result

def multiply_state_dict(state_dict, scalar):
	result = {}
	for key in state_dict.keys():
		result[key] = state_dict[key] * scalar
	return result


# initialize N number of agents (differs for each metaheuristic, TabuSearch has only 1 agent, Simulated Annealing has usually 1, ACO has many, etc.)
num_qubits = 2
num_layers = 4

num_agents = 20
agents = return_random_agents(num_agents, num_qubits, num_layers)

# PSO parameters
num_iterations = 12002
num_particles = num_agents

"""
### GRID SEARCH
inertia_weight_options = [0.6, 0.8, 0.95]  
cognitive_weight_options = [1.25, 1.75, 2.5]  
social_weight_options = [1.75, 2.5] 

# Erstelle ein Gitter aller möglichen Kombinationen
param_grid = list(itertools.product(inertia_weight_options, cognitive_weight_options, social_weight_options))

for i in range(3):
	for params in param_grid:
		mean_reward_list = []
		top_reward_list = []
		iter_index = []
		start_time = time.time()
		
		# Lists to keep track of the scores and agents
		raw_scores = []
		raw_scores_selected = []
		best_scores = []
		runtimes = []

		inertia_weight, cognitive_weight, social_weight = params
		
		pso(agents, num_iterations, num_particles, inertia_weight, cognitive_weight, social_weight, raw_scores, raw_scores_selected, best_scores, runtimes, start_time)


"""
# Finale Ausführung für die besten getesteten Hyperparameter

# PSO parameters
inertia_weight = 0.9
cognitive_weight = 1.0
social_weight = 2.0

for i in range(5):
	mean_reward_list = []
	top_reward_list = []
	iter_index = []
	start_time = time.time()
	
	# Lists to keep track of the scores and agents
	raw_scores = []
	raw_scores_selected = []
	best_scores = []
	runtimes = []
	
	pso(agents, num_iterations, num_particles, inertia_weight, cognitive_weight, social_weight, raw_scores, raw_scores_selected, best_scores, runtimes, start_time)


