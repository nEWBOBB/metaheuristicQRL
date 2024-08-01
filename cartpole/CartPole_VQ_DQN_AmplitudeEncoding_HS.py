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

def mutateold(agent):
	child_agent_var_Q_circuit = agent[0].clone().detach()
	child_agent_var_Q_bias = agent[1].clone().detach()

	
	mutation_power = 0.02 #hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf
			
	child_agent_var_Q_circuit += mutation_power * torch.randn_like(child_agent_var_Q_circuit)
	child_agent_var_Q_bias += mutation_power * torch.randn_like(child_agent_var_Q_bias)

	child_agent = (child_agent_var_Q_circuit, child_agent_var_Q_bias)

	return child_agent

def mutate(agent, bandwidth):
	child_agent_var_Q_circuit = agent[0].clone().detach()
	child_agent_var_Q_bias = agent[1].clone().detach()

	
	mutation_power = 0.02 #hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf
	if random.random() < bandwidth:
			child_agent_var_Q_circuit += mutation_power * torch.randn_like(child_agent_var_Q_circuit)
			child_agent_var_Q_bias += mutation_power * torch.randn_like(child_agent_var_Q_bias)					
	child_agent = (child_agent_var_Q_circuit, child_agent_var_Q_bias)

	return child_agent


def saveresults(iteration, best_scores, raw_scores, raw_scores_selected, runtimes, harmony_memory_size, harmony_memory_considering_rate, pitch_adjusting_rate, bandwidth):
	best_scores = [score[0] for score in best_scores]
	raw_scores = [score[0] for score in raw_scores]
	raw_scores_selected = [score[0] for score in raw_scores_selected]	

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
	results_dir = results_dir + '/CartPole_VQ_DQN_HS'
	os.makedirs(results_dir, exist_ok=True)

	# Erstelle den Dateinamen mit dem Zeitstempel
	csv_filename = f'CartPole_VQ_DQN_HS_HarmonySize_{harmony_memory_size}_ConsideringRate_{harmony_memory_considering_rate}_Pitch_{pitch_adjusting_rate}_bandwidth_{bandwidth}_ITER_{iteration}_TIME_{timestamp}.csv'
	csv_path = os.path.join(results_dir, csv_filename)

	df.to_csv(csv_path, index=False)

def harmony_search(current_agent, current_score, best_agent, best_score, max_iterations, raw_scores, raw_scores_selected, best_scores, runtimes, start_time, harmony_memory_size, harmony_memory_considering_rate, pitch_adjusting_rate, bandwidth, num_qubits, num_layers):
	iteration = 0
	print("Starting Harmony Search")
	# Initialize harmony memory
	harmony_memory = [current_agent]
	harmony_memory_scores = [current_score]

	runtime = 0.0
	max_runtime = 20000.0 # ca 5 hours
	iteration = 0

	while runtime < max_runtime:
		print("ITERATION: ", iteration)

		# Generate a new harmony
		if random.random() < harmony_memory_considering_rate:
			# Select an existing harmony from memory
			selected_index = random.randint(0, len(harmony_memory) - 1)
			new_agent = copy.deepcopy(harmony_memory[selected_index])

			# Adjust the pitch of the selected harmony
			if random.random() < pitch_adjusting_rate:
				new_agent = mutate(new_agent, bandwidth)
		else:
			# Generate a new harmony randomly
			#new_agent = return_random_agents(1)[0]
			new_agent = return_random_agents(1, num_qubits, num_layers)[0]

		# Evaluate the new harmony
		new_score = run_agent_n_times(new_agent, 3)
		print("New Score: ", new_score)

		# Update harmony memory
		if len(harmony_memory) < harmony_memory_size:
			harmony_memory.append(new_agent)
			harmony_memory_scores.append(new_score)
		else:
			worst_index = harmony_memory_scores.index(min(harmony_memory_scores))
			if new_score > harmony_memory_scores[worst_index]:
				harmony_memory[worst_index] = new_agent
				harmony_memory_scores[worst_index] = new_score

		# Update the current and best solutions
		if new_score > current_score:
			current_agent = new_agent
			current_score = new_score
			print("New Current Score: ", new_score)
			raw_scores_selected.append(new_score)
		else:
			raw_scores_selected.append(current_score)

		if new_score > best_score:
			best_agent = new_agent
			best_score = new_score
			print("New Best Score: ", new_score)

		print("Current Best Score: ", best_score)
		print("--------------------")
		raw_scores.append(new_score)
		best_scores.append(best_score)
		runtimes.append(time.time() - start_time)

		print("Runtime: ", runtimes[-1])
		runtime = runtimes[-1]

		# Save the results every 50 iterations
		if iteration % 5000 == 0:
			saveresults(iteration, best_scores, raw_scores, raw_scores_selected, runtimes, harmony_memory_size, harmony_memory_considering_rate, pitch_adjusting_rate, bandwidth)

		iteration += 1
	saveresults(iteration, best_scores, raw_scores, raw_scores_selected, runtimes, harmony_memory_size, harmony_memory_considering_rate, pitch_adjusting_rate, bandwidth)
	return current_agent, current_score, best_agent, best_score, raw_scores, raw_scores_selected, best_scores, runtimes

# initialize N number of agents (differs for each metaheuristic, TabuSearch has only 1 agent, Simulated Annealing has usually 1, ACO has many, etc.)
num_qubits = 2
num_layers = 4

num_agents = 1
agents = return_random_agents(num_agents, num_qubits, num_layers)
"""
###GRID SEARCH
# HS Parameter
harmony_memory_size_options = [50, 100, 150]  
harmony_memory_considering_rate_options = [0.9, 0.65]  
pitch_adjusting_rate_options = [0.7, 0.8]  
bandwidth_options = [0.2, 0.4]  

# Erstelle das Gitter aller möglichen Kombinationen der Hyperparameter für Harmony Search
param_grid = list(itertools.product(
	harmony_memory_size_options, 
	harmony_memory_considering_rate_options, 
	pitch_adjusting_rate_options, 
	bandwidth_options))

for i in range(3):
	for params in param_grid:
		mean_reward_list = []
		top_reward_list = []
		iter_index = []
		start_time = time.time()
		# Agent and Score initialisation
		starting_agent = agents[0]
		starting_agent_score = run_agent_n_times(starting_agent, 3)
		current_agent = copy.deepcopy(starting_agent)
		current_score = starting_agent_score
		best_agent = copy.deepcopy(starting_agent)
		best_score = current_score
		# Lists to keep track of the scores and agents
		raw_scores = [starting_agent_score]
		raw_scores_selected = [starting_agent_score]
		best_scores = [starting_agent_score]
		runtimes = [time.time() - start_time]

		harmony_memory_size, harmony_memory_considering_rate, pitch_adjusting_rate, bandwidth = params
		
		# Iteration variables
		max_iterations = 20002

		harmony_search(current_agent, current_score, best_agent, best_score, max_iterations, raw_scores, raw_scores_selected, best_scores, runtimes, start_time, harmony_memory_size, harmony_memory_considering_rate, pitch_adjusting_rate, bandwidth, num_qubits, num_layers)

"""

# Harmony Search parameters
harmony_memory_size = 100
harmony_memory_considering_rate = 0.8
pitch_adjusting_rate = 0.5
bandwidth = 0.3

for i in range(5):
	mean_reward_list = []
	top_reward_list = []
	iter_index = []
	start_time = time.time()
	# Agent and Score initialisation
	starting_agent = agents[0]
	starting_agent_score = run_agent_n_times(starting_agent, 3)
	current_agent = copy.deepcopy(starting_agent)
	current_score = starting_agent_score
	best_agent = copy.deepcopy(starting_agent)
	best_score = current_score
	# Lists to keep track of the scores and agents
	raw_scores = [starting_agent_score]
	raw_scores_selected = [starting_agent_score]
	best_scores = [starting_agent_score]
	runtimes = [time.time() - start_time]
	# Iteration variables
	max_iterations = 10002
	harmony_search(current_agent, current_score, best_agent, best_score, max_iterations, raw_scores, raw_scores_selected, best_scores, runtimes, start_time, harmony_memory_size, harmony_memory_considering_rate, pitch_adjusting_rate, bandwidth, num_qubits, num_layers)




