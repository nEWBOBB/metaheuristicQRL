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
from metaquantum.CircuitComponents import *
from metaquantum import Optimization

## VQC tools
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
	   
	def get_angles_atan(self, in_x):
		return torch.stack([torch.stack([torch.atan(item), torch.atan(item**2)]) for item in in_x])
   
	def forward(self, batch_item):
		vqc_primitive.var_Q_circuit = self.q_params
		score_batch = []
		for single_item in batch_item:
			res_temp = self.get_angles_atan(single_item)
			q_out_elem = vqc_primitive.forward(res_temp).type(dtype)
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
		#print("x nach MPS: ", x) #tensor([[ 0.0354,  0.0151,  0.0018,  0.0088, -0.0131,  0.0346, -0.0074,  0.0033]])
		x = self.vqc(x)
		#print("x nach VQC: ", x) #tensor([[-0.0038, -0.0136, -0.0091, -0.0081,  0.0062, -0.0319]], dtype=torch.float64)
		x = self.out(x)
		#print("x nach Softmax: ", x) #tensor([[0.1677, 0.1660, 0.1668, 0.1670, 0.1694, 0.1631]], dtype=torch.float64)
		return x

def init_weights(m):
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

def mutate(agent, reference_agent):
	child_agent = copy.deepcopy(agent)
   
	mutation_power = 0.01
	for param, ref_param in zip(child_agent.parameters(), reference_agent.parameters()):
		if len(param.shape) == 4:  # Gewichte von Conv2D
			for i in range(param.shape[0]):
				for j in range(param.shape[1]):
					for k in range(param.shape[2]):
						for l in range(param.shape[3]):
							if np.random.rand() < 0.1:
								param[i][j][k][l] = ref_param[i][j][k][l] + mutation_power * np.random.randn()
		elif len(param.shape) == 3:  # Gewichte von MPS
			for i in range(param.shape[0]):
				for j in range(param.shape[1]):
					for k in range(param.shape[2]):
						if np.random.rand() < 0.1:
							param[i][j][k] = ref_param[i][j][k] + mutation_power * np.random.randn()
		elif len(param.shape) == 2:  # Gewichte von linearer Schicht
			for i in range(param.shape[0]):
				for j in range(param.shape[1]):
					if np.random.rand() < 0.1:
						param[i][j] = ref_param[i][j] + mutation_power * np.random.randn()
		elif len(param.shape) == 1:  # Bias von linearer Schicht oder Conv-Schicht
			for i in range(param.shape[0]):
				if np.random.rand() < 0.1:
					param[i] = ref_param[i] + mutation_power * np.random.randn()
   
	return child_agent

def saveresults(iteration, best_scores, raw_scores, raw_scores_selected, runtimes, num_ants, evaporation_rate, alpha, beta):
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
	results_dir = results_dir + '/5x5_TNVQC_ACO'
	os.makedirs(results_dir, exist_ok=True)

	# Erstelle den Dateinamen mit dem Zeitstempel
	csv_filename = f'5x5_TNVQC_ACO_ants_{num_ants}_evaporation_{evaporation_rate}_alpha_{alpha}_beta_{beta}_Iter_{iteration}_TIME{timestamp}.csv'
	csv_path = os.path.join(results_dir, csv_filename)
	df.to_csv(csv_path, index=False)
   
def ant_colony_optimization(agents, num_iterations, num_ants, evaporation_rate, alpha, beta, raw_scores, raw_scores_selected, best_scores, runtimes, start_time):
	num_agents = len(agents)
	pheromone_matrix = np.ones((num_agents, num_agents))
	best_agent = None
	best_score = -float('inf')
	iteration = 0
	
	while iteration < num_iterations:
		print("ITERATION: ", iteration)

		# Ameisen generieren neue Lösungen basierend auf der Pheromonmatrix
		new_agents = []

		for _ in range(num_ants):
			new_agent = copy.deepcopy(random.choice(agents))
			for _ in range(num_agents):
				agent_scores = [run_agent_n_times(agent, 3)[0] for agent in agents]
			   
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
	   
		# Speichere die Ergebnisse alle 200 Iterationen
		if iteration % 200 == 0:
			saveresults(iteration, best_scores, raw_scores, raw_scores_selected, runtimes, num_ants, evaporation_rate, alpha, beta)

		iteration += 1
	return best_agent, best_score, raw_scores, raw_scores_selected, best_scores, runtimes

#######################
game_actions = 6 

#disable gradients as we will not use them
torch.set_grad_enabled(False)

# initialize N number of agents (differs for each metaheuristic, TabuSearch has only 1 agent, Simulated Annealing has usually 1, ACO has many, etc.)
num_agents = 2
agents = return_random_agents(num_agents)

"""
###Grid Search
# Parameter-Optionen
num_ants_options = [10, 20, 30]  
evaporation_rate_options = [0.9]  
alpha_options = [0.5, 1.0]  
beta_options = [2.0, 3.0] 

# Erstelle ein Gitter aller möglichen Kombinationen
param_grid = list(itertools.product(num_ants_options, evaporation_rate_options, alpha_options, beta_options))

for i in range(3):
	for params in param_grid:
		mean_reward_list = []
		top_reward_list = []
		iter_index = []
		start_time = time.time()
	
		# Listen zum Verfolgen der Ergebnisse
		raw_scores = []
		raw_scores_selected = []
		best_scores = []
		runtimes = []

		num_iterations = 502

		num_ants, evaporation_rate, alpha, beta = params

		ant_colony_optimization(agents, num_iterations, num_ants, evaporation_rate, alpha, beta, raw_scores, raw_scores_selected, best_scores, runtimes, start_time)

"""
# Finale Ausführung für die besten getesteten Hyperparameter
# ACO Parameter
num_iterations = 602
num_ants = 10
evaporation_rate = 0.9
alpha = 0.5
beta = 0.3

for i in range(5):
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
