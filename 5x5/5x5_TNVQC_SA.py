import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

import pandas as pd
import random
import itertools

from datetime import datetime
import pickle
from collections import deque

from gym.wrappers import Monitor
from gym_minigrid.wrappers import *
from MiniGridWrappers import ImgObsFlatWrapper

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PyTensorNet.torchmps import MPS

import math
import copy
import os

# metaQuantum
from metaquantum.CircuitComponents import *
from metaquantum import Optimization

## VQC tools
# qdevice = "default.qubit"
qdevice = "qulacs.simulator"

gpu_q = False
ACTION_DIM = 6
vqc_primitive = VariationalQuantumClassifierInterBlock_M_IN_N_OUT(
	num_of_input= 8,
	num_of_output= ACTION_DIM,
	num_of_wires = 8,
	num_of_layers = 1,
	qdevice = qdevice,
	hadamard_gate = True,
	more_entangle = False,
	gpu = gpu_q
	)

class VQCTorch(nn.Module):
	def __init__(self):
		super().__init__()
		self.q_params = nn.Parameter(0.01 * torch.randn(1, 8, 3))
		#DEBUG
		#print("Die VQC Parameter: ", self.q_params)

	def get_angles_atan(self, in_x):
		return torch.stack([torch.stack([torch.atan(item), torch.atan(item**2)]) for item in in_x])

	def forward(self, batch_item):

		vqc_primitive.var_Q_circuit = self.q_params
		# print(vqc.var_Q_circuit.dtype)
		score_batch = []

		for single_item in batch_item:
			res_temp = self.get_angles_atan(single_item)
			# print(res_temp)

			q_out_elem = vqc_primitive.forward(res_temp).type(dtype)

			# print(q_out_elem)
		
			# clamp = 1e-9 * torch.ones(2).type(dtype).to(device)
			# clamp = 1e-9 * torch.ones(2)
			# normalized_output = torch.max(q_out_elem, clamp)
			# score_batch.append(normalized_output)
			score_batch.append(q_out_elem)

		scores = torch.stack(score_batch).view(len(batch_item),ACTION_DIM)

		return scores


class CartPoleAI(nn.Module):
	def __init__(self, mps = None):
		super().__init__()
		if mps != None:
			self.mps = mps
		else:
			self.mps = MPS(input_dim = 147, output_dim = 8, bond_dim = 1, feature_dim = 2, use_GPU = False, parallel = True, init_std=1e-2)
		self.vqc = VQCTorch()
		self.out = nn.Softmax(dim=1)

			
	def forward(self, inputs):
		x = self.mps(inputs)
		x = self.vqc(x)
		x = self.out(x)
		return x


def init_weights(m):
		# nn.Conv2d weights are of shape [16, 1, 3, 3] i.e. # number of filters, 1, stride, stride
		# nn.Conv2d bias is of shape [16] i.e. # number of filters
		
		# nn.Linear weights are of shape [32, 24336] i.e. # number of input features, number of output features
		# nn.Linear bias is of shape [32] i.e. # number of output features
		
	if ((type(m) == nn.Linear) | (type(m) == nn.Conv2d)):
		torch.nn.init.xavier_uniform(m.weight)
		m.bias.data.fill_(0.00)


def return_random_agents(num_agents):
	
	agents = []
	for _ in range(num_agents):
		
		agent = CartPoleAI()
		
		for param in agent.parameters():
			param.requires_grad = False
			
		init_weights(agent)
		agents.append(agent)
		
		
	return agents


def run_agents(agents):
	
	reward_agents = []
	env = gym.make('MiniGrid-Empty-5x5-v0')
	env = ImgObsFlatWrapper(env)
	
	for agent in agents:
		agent.eval()
	
		observation = env.reset()
		
		r=0
		s=0
		
		for _ in range(5000):
			
			inp = torch.tensor(observation).type('torch.FloatTensor').view(1,-1)
			output_probabilities = agent(inp).cpu().detach().numpy()[0]
			action = np.random.choice(range(game_actions), 1, p=output_probabilities).item()
			new_observation, reward, done, info = env.step(action)
			r=r+reward
			
			s=s+1
			observation = new_observation

			if(done):
				break

		reward_agents.append(r)        
		#reward_agents.append(s)
		
	
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

	child_agent = copy.deepcopy(agent)
	
	mutation_power = 0.01 #hyper-parameter, set from https://arxiv.org/pdf/1712.06567.pdf

	# Need to modify for the VQC parameters. The param shape is not the same.
	#print("Die Child Agent Parameter: ", child_agent.parameters())
	   		
	for param in child_agent.parameters():
	
		if(len(param.shape)==4): #weights of Conv2D

			for i0 in range(param.shape[0]):
				for i1 in range(param.shape[1]):
					for i2 in range(param.shape[2]):
						for i3 in range(param.shape[3]):
							
							param[i0][i1][i2][i3]+= mutation_power * np.random.randn()
								
		elif(len(param.shape)==3): #weights of MPS

			for i0 in range(param.shape[0]):
				for i1 in range(param.shape[1]):
					for i2 in range(param.shape[2]):
						param[i0][i1][i2]+= mutation_power * np.random.randn()							

		elif(len(param.shape)==2): #weights of linear layer
			for i0 in range(param.shape[0]):
				for i1 in range(param.shape[1]):
					
					param[i0][i1]+= mutation_power * np.random.randn()
						

		elif(len(param.shape)==1): #biases of linear layer or conv layer
			for i0 in range(param.shape[0]):
				
				param[i0]+=mutation_power * np.random.randn()

		else:
			print("NO PARAM MUTATION!")

	return child_agent


def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	return np.exp(x) / np.sum(np.exp(x), axis=0)


def play_agent(agent):
	try: #try and exception block because, render hangs if an erorr occurs, we must do env.close to continue working    
		env = gym.make('MiniGrid-Empty-5x5-v0')
		env = ImgObsFlatWrapper(env)

		
		env_record = Monitor(env, './video', force=True)
		observation = env_record.reset()
		last_observation = observation
		r=0
		for _ in range(1000):
			env_record.render()
			inp = torch.tensor(observation).type('torch.FloatTensor').view(1,-1)
			output_probabilities = agent(inp).detach().numpy()[0]
			action = np.random.choice(range(game_actions), 1, p=output_probabilities).item()
			new_observation, reward, done, info = env_record.step(action)
			r=r+reward
			observation = new_observation

			if(done):
				break

		env_record.close()
		print("Rewards: ",r)

	except Exception as e:
		env_record.close()
		print(e.__doc__)
		print(str(e))


def saveresults(iteration, start_temperature, best_scores, raw_scores, raw_scores_selected, runtimes):
	best_scores = [score[0] for score in best_scores]
	raw_scores = [score[0] for score in raw_scores]
	raw_scores_selected = [score[0] for score in raw_scores_selected]	

	iter_list = list(range(1, len(raw_scores) + 1))

	assert len(raw_scores) == len(raw_scores_selected) == len(best_scores)
	# Erstelle ein DataFrame mit den Daten
	data = {"raw": raw_scores, "rawselected": raw_scores_selected, "best": best_scores, "generations": iter_list, "temperature": temperature, "runtime": runtimes}
	df = pd.DataFrame(data)

	# Speichere das DataFrame in einer CSV-Datei im Ordner "results" im selben Verzeichnis wie das Skript
	script_dir = os.path.dirname(os.path.abspath(__file__))

	# Erhalte den aktuellen Zeitpunkt und formatiere ihn als String
	now = datetime.now()
	timestamp = now.strftime('%m-%d_%H-%M-%S')

	# Erstelle den Ordner "results", wenn er noch nicht existiert
	results_dir = os.path.join(script_dir, 'results')
	results_dir = results_dir + '/5x5_TNVQC_SA'
	os.makedirs(results_dir, exist_ok=True)

	# Erstelle den Dateinamen mit dem Zeitstempel
	csv_filename = f'5x5_TNVQC_SA_StartTemp_{start_temperature}_Cool_{cooling_rate}_Iter_{iteration}_TIME{timestamp}.csv'
	csv_path = os.path.join(results_dir, csv_filename)

	df.to_csv(csv_path, index=False)

def simulated_annealing(current_agent, current_score, best_agent, best_score, temperature, cooling_rate, max_iterations, raw_scores, raw_scores_selected, best_scores, runtimes, start_time):
	iteration = 0
	start_temperature = temperature
	print("Starting Simulated Annealing")
	while temperature > 0.001 and iteration < max_iterations:
	#while iteration < max_iterations:
		# Generate a new candidate solution by perturbing the current parameters
		print("ITERATION: ", iteration)
		print("Temperature: ", temperature)
		new_agent = copy.deepcopy(current_agent)
		new_agent = mutate(new_agent)

		# Evaluate the candidate solution
		new_score = run_agent_n_times(new_agent, 3)

		# Determine whether to accept the candidate solution
		print("Current Score: ", current_score)
		print("New Score: ", new_score)
		acceptance_probability = math.exp((new_score[0] - current_score[0]) / temperature)
		print("Acceptance Probability: ", acceptance_probability)
		if acceptance_probability > random.random():
			current_agent = new_agent
			current_score = new_score
			print("Accepted! New Score: ", new_score)
			raw_scores_selected.append(new_score)
		else:
			print("NOT ACCEPTED")
			raw_scores_selected.append(current_score)

		# Update the best solution if necessary
		if new_score[0] > best_score[0]:
			best_agent = new_agent
			best_score = new_score
			print("New Best Score: ", new_score)

		print("--------------------")
		raw_scores.append(new_score)
		best_scores.append(best_score)
		runtimes.append(time.time() - start_time)

		# Save the results every 10 iterations
		if iteration % 1000 == 0:
			saveresults(iteration, start_temperature, best_scores, raw_scores, raw_scores_selected, runtimes)

		# Cool down the temperature
		temperature *= 1 - cooling_rate
		iteration += 1

	return current_agent, current_score, best_agent, best_score, raw_scores, raw_scores_selected, best_scores, runtimes

#######################
game_actions = 6

#disable gradients as we will not use them
torch.set_grad_enabled(False)

# initialize N number of agents (differs for each metaheuristic, TabuSearch has only 1 agent, Simulated Annealing has usually 1, ACO has many, etc.)
num_agents = 1
agents = return_random_agents(num_agents)
"""
### GRID SEARCH
# Definiere mögliche Werte für jeden Hyperparameter
temperatures = [100.0, 1000.0] #[1.0, 10.0, 50.0]
cooling_rates = [0.01, 0.005, 0.001] #[0.01, 0.05, 0.1]

# Erstelle ein Gitter aller möglichen Kombinationen
param_grid = list(itertools.product(temperatures, cooling_rates))


# Führe den Algorithmus für jede Kombination aus
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

		# Iteration variables
		max_iterations = 10001

		temperature, cooling_rate = params

		simulated_annealing(current_agent, current_score, best_agent, best_score, temperature, cooling_rate, max_iterations, raw_scores, raw_scores_selected, best_scores, runtimes, start_time)



"""
# Finale Ausführung für die besten getesteten Hyperparameter
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
	max_iterations = 25002

	temperature = 3000.0
	cooling_rate = 0.001

	simulated_annealing(current_agent, current_score, best_agent, best_score, temperature, cooling_rate, max_iterations, raw_scores, raw_scores_selected, best_scores, runtimes, start_time)
	
	#for i in range(3):
		#play_agent(best_agent)

