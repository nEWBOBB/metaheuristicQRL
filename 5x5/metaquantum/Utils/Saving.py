import pickle
import torch

from .Plotting import plotTrainingResult

def save_test():
	file_title = "TEST_SAVING"
	with open(file_title + ".txt", "wb") as fp:
			pickle.dump("training_x", fp)


def save_training_and_testing(exp_name, file_title, training_x, training_y, val_x, val_y, testing_x, testing_y):
	file_title = exp_name + "/" + file_title + "_DATASET_"
	with open(file_title + "_TRAINING_X" + ".txt", "wb") as fp:
			pickle.dump(training_x, fp)
	with open(file_title + "_TRAINING_Y" + ".txt", "wb") as fp:
			pickle.dump(training_y, fp)
	with open(file_title + "_VAL_X" + ".txt", "wb") as fp:
			pickle.dump(val_x, fp)
	with open(file_title + "_VAL_Y" + ".txt", "wb") as fp:
			pickle.dump(val_y, fp)
	with open(file_title + "_TESTING_X" + ".txt", "wb") as fp:
			pickle.dump(testing_x, fp)
	with open(file_title + "_TESTING_Y" + ".txt", "wb") as fp:
			pickle.dump(testing_y, fp)

## Ploting Function ##

def saving_torch_model(exp_name, file_title, iter_count, var_Q_circuit):
	file_title = exp_name + "/" + file_title + "_Iter_Count_" + str(iter_count[-1])
	# torch.save(model.state_dict(), PATH)
	torch.save(var_Q_circuit, file_title + "_torch_model_dict" + ".pth")


def save_all_the_current_info(exp_name, file_title, iter_count, var_Q_circuit, var_Q_bias, cost_train, cost_test, acc_train, acc_val, acc_test):
	## Saving the model
	file_title = exp_name + "/" + file_title + "_Iter_Count_" + str(iter_count[-1])

	with open(file_title + "_var_Q_circuit" + ".txt", "wb") as fp:
			pickle.dump(var_Q_circuit, fp)

	with open(file_title + "_var_Q_bias" + ".txt", "wb") as fp:
			pickle.dump(var_Q_bias, fp)

	with open(file_title + "_cost_train" + ".txt", "wb") as fp:
			pickle.dump(cost_train, fp)

	with open(file_title + "_cost_test" + ".txt", "wb") as fp:
			pickle.dump(cost_test, fp)

	with open(file_title + "_acc_train" + ".txt", "wb") as fp:
			pickle.dump(acc_train, fp)

	with open(file_title + "_acc_val" + ".txt", "wb") as fp:
			pickle.dump(acc_val, fp)

	with open(file_title + "_acc_test" + ".txt", "wb") as fp:
			pickle.dump(acc_test, fp)
 
	plotTrainingResult(iter_count, cost_train, cost_test, acc_train, acc_val, acc_test, exp_name, file_title)