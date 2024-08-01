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

def mutate(agent, reference_agent):
	child_agent_var_Q_circuit = agent[0].clone().detach()
	child_agent_var_Q_bias = agent[1].clone().detach()

	reference_agent_var_Q_circuit = reference_agent[0].clone().detach()
	reference_agent_var_Q_bias = reference_agent[1].clone().detach()
	
	mutation_power = 0.02 #hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf
					
	child_agent_var_Q_circuit += mutation_power * torch.randn_like(reference_agent_var_Q_circuit)
	child_agent_var_Q_bias += mutation_power * torch.randn_like(reference_agent_var_Q_bias)	

	child_agent = (child_agent_var_Q_circuit, child_agent_var_Q_bias)

	return child_agent


def saveresults(iteration, num_agents, best_scores, raw_scores, raw_scores_selected, runtimes, num_ants, evaporation_rate, alpha, beta):
	iter_list = list(range(1, len(best_scores) + 1))
	data = {"best": best_scores, "generations": iter_list, "runtime": runtimes}
	df = pd.DataFrame(data)

	# Speichere das DataFrame in einer CSV-Datei im Ordner "results" im selben Verzeichnis wie das Skript
	script_dir = os.path.dirname(os.path.abspath(__file__))

	# Erhalte den aktuellen Zeitpunkt und formatiere ihn als String
	now = datetime.now()
	timestamp = now.strftime('%m-%d_%H-%M-%S')

	# Erstelle den Ordner "results", wenn er noch nicht existiert
	results_dir = os.path.join(script_dir, 'results')
	results_dir = results_dir + '/CartPole_VQ_DQN_ACO'
	os.makedirs(results_dir, exist_ok=True)

	# Erstelle den Dateinamen mit dem Zeitstempel
	csv_filename = f'CartPole_VQ_DQN_SA_ACO_agents_{num_agents}_ants_{num_ants}_evaporation_{evaporation_rate}_alpha_{alpha}_beta_{beta}_Iter_{iteration}_TIME{timestamp}.csv'
	csv_path = os.path.join(results_dir, csv_filename)
	df.to_csv(csv_path, index=False)


def ant_colony_optimization(agents, num_iterations, num_ants, evaporation_rate, alpha, beta, raw_scores, raw_scores_selected, best_scores, runtimes, start_time):
	num_agents = len(agents)
	pheromone_matrix = np.ones((num_agents, num_agents))
	best_agent = None
	best_score = -float('inf')

	runtime = 0.0
	max_runtime = 30000.0 # ca 5 hours
	iteration = 0

	while runtime < max_runtime:
	#while iteration < num_iterations:
		print("ITERATION: ", iteration)

		# Ameisen generieren neue Lösungen basierend auf der Pheromonmatrix
		new_agents = []

		for _ in range(num_ants):
			new_agent = copy.deepcopy(random.choice(agents))
			for _ in range(num_agents):
				agent_scores = [run_agent_n_times(agent, 2)[0] for agent in agents]
			   
				# Verschiebe Bewertungen in den positiven Bereich
				min_score = min(agent_scores)
				agent_scores = [score - min_score + 1e-5 for score in agent_scores]
			   
				probabilities = np.power(pheromone_matrix[_], alpha) * np.power(agent_scores, beta)
				probabilities_sum = np.sum(probabilities)
			   
				# Überprüfen ob die Summe der Wahrscheinlichkeiten größer als Null ist
				if probabilities_sum > 0:
					probabilities /= probabilities_sum
				else:
					# Standardwerte wenn die Summe Null ist
					probabilities = np.ones(num_agents) / num_agents
			   
				selected_index = np.random.choice(range(num_agents), p=probabilities)
				new_agent = mutate(new_agent, agents[selected_index])
			new_agents.append(new_agent)
	   
		# Bewerte die neuen Lösungen
		scores = run_agents_n_times(new_agents, 3)
	   
		# Aktualisiere die beste Lösung
		for agent, score in zip(new_agents, scores):
			if score > best_score:
				best_agent = agent
				best_score = score
				print("NEW BEST SCORE: ", best_score)
			else:
				print("Score: ", score)

			print("Best score: ", best_score)

		# Aktualisiere die Pheromonmatrix
		pheromone_matrix *= evaporation_rate
		for i in range(num_agents):
			for j in range(num_agents):
				pheromone_matrix[i][j] += sum(scores) / len(scores)
	   
		# Speichere die Ergebnisse
		raw_scores.extend(scores)
		raw_scores_selected.append(best_score)
		best_scores.append(best_score)
		
		runtimes.append(time.time() - start_time)
	   
		print("Runtime: ", runtimes[-1])
		runtime = runtimes[-1]
		
		# Speichere die Ergebnisse alle 50 Iterationen
		if iteration % 400 == 0:
			saveresults(iteration, num_agents, best_scores, raw_scores, raw_scores_selected, runtimes, num_ants, evaporation_rate, alpha, beta)

		iteration += 1
	saveresults(iteration, num_agents, best_scores, raw_scores, raw_scores_selected, runtimes, num_ants, evaporation_rate, alpha, beta)
	return best_agent, best_score, raw_scores, raw_scores_selected, best_scores, runtimes

# initialize N number of agents (differs for each metaheuristic, TabuSearch has only 1 agent, Simulated Annealing has usually 1, ACO has many, etc.)
num_qubits = 2
num_layers = 4

#num_agents = 2
#agents = return_random_agents(num_agents, num_qubits, num_layers)
"""
###Grid Search
# Parameter-Optionen
num_agents_options = [2, 4, 6]
num_ants_options = [10, 20, 30]  
evaporation_rate_options = [0.85, 0.95]  
alpha_options = [1.0, 1.5]  
beta_options = [1.0, 1.5] 

# Erstelle ein Gitter aller möglichen Kombinationen
param_grid = list(itertools.product(num_agents_options, num_ants_options, evaporation_rate_options, alpha_options, beta_options))

for i in range(3):
	for params in param_grid:
		num_agents, num_ants, evaporation_rate, alpha, beta = params
		agents = return_random_agents(num_agents, num_qubits, num_layers)
		mean_reward_list = []
		top_reward_list = []
		iter_index = []
		start_time = time.time()
	
		# Listen zum Verfolgen der Ergebnisse
		raw_scores = []
		raw_scores_selected = []
		best_scores = []
		runtimes = []

		num_iterations = 801

		ant_colony_optimization(agents, num_iterations, num_ants, evaporation_rate, alpha, beta, raw_scores, raw_scores_selected, best_scores, runtimes, start_time)
"""

# Finale Ausführung für die besten getesteten Hyperparameter
# ACO Parameter
num_iterations = 100002
num_ants = 30
evaporation_rate = 0.95
alpha = 1.0
beta = 1.5
num_agents = 5

for i in range(5):
	agents = return_random_agents(num_agents, num_qubits, num_layers)
	mean_reward_list = []
	top_reward_list = []
	iter_index = []
	start_time = time.time()
   
	# Listen zum Verfolgen der Ergebnisse
	raw_scores = []
	raw_scores_selected = []
	best_scores = []
	runtimes = []
   
	ant_colony_optimization(agents, num_iterations, num_ants, evaporation_rate, alpha, beta, raw_scores, raw_scores_selected, best_scores, runtimes, start_time)





