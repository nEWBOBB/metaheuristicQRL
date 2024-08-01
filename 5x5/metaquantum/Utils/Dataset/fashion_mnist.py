# 2020 08 24
# Fashion MNIST

# 2019 10 05 
# For generate MNIST data from the original 28*28 data
# With possible downsampling
# Binary output for specified two number in 0~9


from keras.datasets import fashion_mnist
import numpy as np 
import torch
import matplotlib.pyplot as plt

###

def plot_two_classes(data_set_category_1, data_set_category_2):

	x = np.arange(1024)

	fig, ax = plt.subplots(3,1)
	class_1 = ax[0].bar(x, data_set_category_1, color='b')
	class_2 = ax[0].bar(x, data_set_category_2, color='r')

	class_a = ax[1].bar(x, data_set_category_1, color='b')
	class_b = ax[2].bar(x, data_set_category_2, color='r')

	plt.show()
	
	return

def display_img(image):
	# original = np.transpose(original, (1, 2, 0))
	# adversarial = np.transpose(adversarial, (1, 2, 0))

	plt.figure()

	plt.title('Image')
	plt.imshow(image)  # division by 255 to convert [0, 255] to [0, 1]
	plt.axis('off')

	plt.show()

## Data Preprocessing ##


## Data Output ##

def torch_data_loading():
	pass

def down_sampling():
	pass

def get_target_num():
	pass


def data_loading_down_sampled():
	(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
	# Normalized
	x_train = x_train[:,5:23,5:23]
	x_test = x_test[:,5:23,5:23]

	# x_train = (x_train.reshape(60000, 784)/255. - 0.1307)/0.3081
	# x_test = (x_test.reshape(10000, 784)/255. - 0.1307)/0.3081

	x_train = x_train.reshape(60000, 324)/255. 
	x_test = x_test.reshape(10000, 324)/255. 

	x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
	x_test = torch.from_numpy(x_test).type(torch.FloatTensor)

	y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_test = torch.from_numpy(y_test).type(torch.LongTensor)

	return x_train, y_train, x_test, y_test

def data_loading():
	(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
	# Normalized

	x_train = x_train.reshape(60000, 784)/255.
	x_test = x_test.reshape(10000, 784)/255.

	x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
	x_test = torch.from_numpy(x_test).type(torch.FloatTensor)

	y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_test = torch.from_numpy(y_test).type(torch.LongTensor)

	return x_train, y_train, x_test, y_test

def data_loading_with_padding():
	(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

	x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant')
	x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'constant')
	# Normalized

	x_train = x_train.reshape(60000, 1024)/255.
	x_test = x_test.reshape(10000, 1024)/255.

	x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
	x_test = torch.from_numpy(x_test).type(torch.FloatTensor)

	y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_test = torch.from_numpy(y_test).type(torch.LongTensor)

	return x_train, y_train, x_test, y_test

def data_loading_with_target(target_num):
	# Should clean out the test data set also!!!!!!

	(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

	# Normalized

	x_train = x_train.reshape(60000, 784)/255.
	x_test = x_test.reshape(10000, 784)/255.

	# Select out the train target

	x_true_train = []
	y_true_train = []

	for idx in range(len(y_train)):
		if y_train[idx] == target_num:
			x_true_train.append(x_train[idx])
			y_true_train.append(y_train[idx])

	x_true_train = np.array(x_true_train)
	y_true_train = np.array(y_true_train)

	# Select out the test target

	x_true_test = []
	y_true_test = []

	for idx in range(len(y_test)):
		if y_test[idx] == target_num:
			x_true_test.append(x_test[idx])
			y_true_test.append(y_test[idx])

	x_true_train = np.array(x_true_train)
	y_true_train = np.array(y_true_train)

	x_true_test = np.array(x_true_test)
	y_true_test = np.array(y_true_test)

	# x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
	x_train = torch.from_numpy(x_true_train).type(torch.FloatTensor)
	x_test = torch.from_numpy(x_true_test).type(torch.FloatTensor)

	# y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_train = torch.from_numpy(y_true_train).type(torch.LongTensor)
	y_test = torch.from_numpy(y_true_test).type(torch.LongTensor)

	return x_train, y_train, x_test, y_test

def data_loading_with_padding_target(target_num):
	(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

	x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant')
	x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'constant')

	# Normalized

	x_train = x_train.reshape(60000, 1024)/255.
	x_test = x_test.reshape(10000, 1024)/255.

	# Select out the train target

	x_true_train = []
	y_true_train = []

	for idx in range(len(y_train)):
		if y_train[idx] == target_num:
			x_true_train.append(x_train[idx])
			y_true_train.append(y_train[idx])

	x_true_train = np.array(x_true_train)
	y_true_train = np.array(y_true_train)

	# Select out the test target

	x_true_test = []
	y_true_test = []

	for idx in range(len(y_test)):
		if y_test[idx] == target_num:
			x_true_test.append(x_test[idx])
			y_true_test.append(y_test[idx])

	x_true_train = np.array(x_true_train)
	y_true_train = np.array(y_true_train)

	x_true_test = np.array(x_true_test)
	y_true_test = np.array(y_true_test)

	# x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
	x_train = torch.from_numpy(x_true_train).type(torch.FloatTensor)
	x_test = torch.from_numpy(x_true_test).type(torch.FloatTensor)

	# y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_train = torch.from_numpy(y_true_train).type(torch.LongTensor)
	y_test = torch.from_numpy(y_true_test).type(torch.LongTensor)

	return x_train, y_train, x_test, y_test

def data_loading_with_padding_full():
	(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

	x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant')
	x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'constant')

	# Normalized

	x_train = x_train.reshape(60000, 1024)/255.
	x_test = x_test.reshape(10000, 1024)/255.

	# x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
	x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
	x_test = torch.from_numpy(x_test).type(torch.FloatTensor)

	# y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_test = torch.from_numpy(y_test).type(torch.LongTensor)

	return x_train, y_train, x_test, y_test


def data_loading_one_target_vs_others(target_num):
	# This function is to output data set as 
	# one-half os the target_num which is about 6000 data point
	# plus the same number randomly selected from the others
	# and the test set is divided following the same method
	# 1000 target_num and 1000 others

	(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

	x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant')
	x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'constant')

	# Normalized

	x_train = x_train.reshape(60000, 1024)/255.
	x_test = x_test.reshape(10000, 1024)/255.

	# Select out the train target
	# Select out the train non-target
	# Split the data

	x_true_train = []
	y_true_train = []
	x_false_train = []
	y_false_train = []

	for idx in range(len(y_train)):
		if y_train[idx] == target_num:
			x_true_train.append(x_train[idx])
			y_true_train.append(y_train[idx])
		else:
			x_false_train.append(x_train[idx])
			y_false_train.append(y_train[idx])

	x_true_train = np.array(x_true_train)
	y_true_train = np.array(y_true_train)
	x_false_train = np.array(x_false_train)
	y_false_train = np.array(y_false_train)

	permutation_false_train = np.random.permutation(len(y_false_train))

	x_false_train = x_false_train[permutation_false_train[:len(y_true_train)]]
	y_false_train = y_false_train[permutation_false_train[:len(y_true_train)]]


	# Split the data in the test set
	# Select out the test target
	# Select out the train non-target
	x_true_test = []
	y_true_test = []
	x_false_test = []
	y_false_test = []

	for idx in range(len(y_test)):
		if y_test[idx] == target_num:
			x_true_test.append(x_test[idx])
			y_true_test.append(y_test[idx])
		else:
			x_false_test.append(x_test[idx])
			y_false_test.append(y_test[idx])

	x_true_test = np.array(x_true_test)
	y_true_test = np.array(y_true_test)
	x_false_test = np.array(x_false_test)
	y_false_test = np.array(y_false_test)

	permutation_false_test = np.random.permutation(len(y_false_test))

	x_false_test = x_false_test[permutation_false_test[:len(y_true_test)]]
	y_false_test = y_false_test[permutation_false_test[:len(y_true_test)]]


	# Combine the data

	permutation_balanced_train = np.random.permutation(len(x_true_train) * 2)
	x_train_balanced = np.concatenate((x_true_train, x_false_train), axis = 0)
	y_train_balanced = np.concatenate((y_true_train, y_false_train), axis = 0)

	x_train_balanced = x_train_balanced[permutation_balanced_train]
	y_train_balanced = y_train_balanced[permutation_balanced_train]

	permutation_balanced_test = np.random.permutation(len(x_true_test) * 2)
	x_test_balanced = np.concatenate((x_true_test, x_false_test), axis = 0)
	y_test_balanced = np.concatenate((y_true_test, y_false_test), axis = 0)

	x_test_balanced = x_test_balanced[permutation_balanced_test]
	y_test_balanced = y_test_balanced[permutation_balanced_test]



	# Transform the y label to 0 or 1
	y_train_balanced = (y_train_balanced == target_num)
	y_test_balanced = (y_test_balanced == target_num)
	
	# x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
	x_train = torch.from_numpy(x_train_balanced).type(torch.FloatTensor)
	x_test = torch.from_numpy(x_test_balanced).type(torch.FloatTensor)

	# y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_train = torch.from_numpy(y_train_balanced).type(torch.LongTensor)
	y_test = torch.from_numpy(y_test_balanced).type(torch.LongTensor)

	

	return x_train, y_train, x_test, y_test


def data_loading_full_target(padding = False, size_for_each_class = 0):
	x_train_0 = None
	y_train_0 = None
	x_test_0 = None
	y_test_0 = None
	x_train_1 = None
	y_train_1 = None
	x_test_1 = None
	y_test_1 = None
	x_train_2 = None
	y_train_2 = None
	x_test_2 = None
	y_test_2 = None
	x_train_3 = None
	y_train_3 = None
	x_test_3 = None
	y_test_3 = None
	x_train_4 = None
	y_train_4 = None
	x_test_4 = None
	y_test_4 = None
	x_train_5 = None
	y_train_5 = None
	x_test_5 = None
	y_test_5 = None
	x_train_6 = None
	y_train_6 = None
	x_test_6 = None
	y_test_6 = None
	x_train_7 = None
	y_train_7 = None
	x_test_7 = None
	y_test_7 = None
	x_train_8 = None
	y_train_8 = None
	x_test_8 = None
	y_test_8 = None
	x_train_9 = None
	y_train_9 = None
	x_test_9 = None
	y_test_9 = None

	if padding == True:
		x_train_0, y_train_0, x_test_0, y_test_0 = data_loading_with_padding_target(0)
		x_train_1, y_train_1, x_test_1, y_test_1 = data_loading_with_padding_target(1)
		x_train_2, y_train_2, x_test_2, y_test_2 = data_loading_with_padding_target(2)
		x_train_3, y_train_3, x_test_3, y_test_3 = data_loading_with_padding_target(3)
		x_train_4, y_train_4, x_test_4, y_test_4 = data_loading_with_padding_target(4)
		x_train_5, y_train_5, x_test_5, y_test_5 = data_loading_with_padding_target(5)
		x_train_6, y_train_6, x_test_6, y_test_6 = data_loading_with_padding_target(6)
		x_train_7, y_train_7, x_test_7, y_test_7 = data_loading_with_padding_target(7)
		x_train_8, y_train_8, x_test_8, y_test_8 = data_loading_with_padding_target(8)
		x_train_9, y_train_9, x_test_9, y_test_9 = data_loading_with_padding_target(9)
	else:
		x_train_0, y_train_0, x_test_0, y_test_0 = data_loading_with_target(0)
		x_train_1, y_train_1, x_test_1, y_test_1 = data_loading_with_target(1)
		x_train_2, y_train_2, x_test_2, y_test_2 = data_loading_with_target(2)
		x_train_3, y_train_3, x_test_3, y_test_3 = data_loading_with_target(3)
		x_train_4, y_train_4, x_test_4, y_test_4 = data_loading_with_target(4)
		x_train_5, y_train_5, x_test_5, y_test_5 = data_loading_with_target(5)
		x_train_6, y_train_6, x_test_6, y_test_6 = data_loading_with_target(6)
		x_train_7, y_train_7, x_test_7, y_test_7 = data_loading_with_target(7)
		x_train_8, y_train_8, x_test_8, y_test_8 = data_loading_with_target(8)
		x_train_9, y_train_9, x_test_9, y_test_9 = data_loading_with_target(9)

	if size_for_each_class:
		x_train_0 = x_train_0[:size_for_each_class]
		y_train_0 = y_train_0[:size_for_each_class]
		x_train_1 = x_train_1[:size_for_each_class]
		y_train_1 = y_train_1[:size_for_each_class]
		x_train_2 = x_train_2[:size_for_each_class]
		y_train_2 = y_train_2[:size_for_each_class]
		x_train_3 = x_train_3[:size_for_each_class]
		y_train_3 = y_train_3[:size_for_each_class]
		x_train_4 = x_train_4[:size_for_each_class]
		y_train_4 = y_train_4[:size_for_each_class]
		x_train_5 = x_train_5[:size_for_each_class]
		y_train_5 = y_train_5[:size_for_each_class]
		x_train_6 = x_train_6[:size_for_each_class]
		y_train_6 = y_train_6[:size_for_each_class]
		x_train_7 = x_train_7[:size_for_each_class]
		y_train_7 = y_train_7[:size_for_each_class]
		x_train_8 = x_train_8[:size_for_each_class]
		y_train_8 = y_train_8[:size_for_each_class]
		x_train_9 = x_train_9[:size_for_each_class]
		y_train_9 = y_train_9[:size_for_each_class]

	x_train_combined = np.concatenate((x_train_0, x_train_1, x_train_2, x_train_3, x_train_4, x_train_5, x_train_6, x_train_7, x_train_8, x_train_9), axis = 0)
	y_train_combined = np.concatenate((y_train_0, y_train_1, y_train_2, y_train_3, y_train_4, y_train_5, y_train_6, y_train_7, y_train_8, y_train_9), axis = 0)

	x_test_combined = np.concatenate((x_test_0, x_test_1, x_test_2, x_test_3, x_test_4, x_test_5, x_test_6, x_test_7, x_test_8, x_test_9), axis = 0)
	y_test_combined = np.concatenate((y_test_0, y_test_1, y_test_2, y_test_3, y_test_4, y_test_5, y_test_6, y_test_7, y_test_8, y_test_9), axis = 0)

	permutation_train = np.random.permutation(size_for_each_class * 10)
	permutation_test = np.random.permutation(size_for_each_class * 10)

	x_train_combined = x_train_combined[permutation_train]
	y_train_combined = y_train_combined[permutation_train]

	x_test_combined = x_test_combined[permutation_test]
	y_test_combined = y_test_combined[permutation_test]

	# x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
	x_train = torch.from_numpy(x_train_combined).type(torch.FloatTensor)

	# y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_train = torch.from_numpy(y_train_combined).type(torch.LongTensor)

	x_test = torch.from_numpy(x_test_combined).type(torch.FloatTensor)

	# y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_test = torch.from_numpy(y_test_combined).type(torch.LongTensor)

	return x_train, y_train, x_test, y_test



def data_loading_two_target(target_1, target_2, padding = False, size_for_each_class = 0):
	# Define the first data be labeled 0
	# Define the second data be labeled 1
	x_train_first = None
	y_train_first = None
	x_test_first = None
	y_test_first = None
	x_train_second = None
	y_train_second = None
	x_test_second = None
	y_test_second = None

	if padding == True:
		x_train_first, y_train_first, x_test_first, y_test_first = data_loading_with_padding_target(target_1)
		x_train_second, y_train_second, x_test_second, y_test_second = data_loading_with_padding_target(target_2)

	else:
		x_train_first, y_train_first, x_test_first, y_test_first = data_loading_with_target(target_1)
		x_train_second, y_train_second, x_test_second, y_test_second = data_loading_with_target(target_2)

	if size_for_each_class:
		x_train_first = x_train_first[:size_for_each_class]
		y_train_first = y_train_first[:size_for_each_class]
		x_train_second = x_train_second[:size_for_each_class]
		y_train_second = y_train_second[:size_for_each_class]

	x_train_combined = np.concatenate((x_train_first, x_train_second), axis = 0)
	y_train_combined = np.concatenate((y_train_first, y_train_second), axis = 0)

	x_test_combined = np.concatenate((x_test_first, x_test_second), axis = 0)
	y_test_combined = np.concatenate((y_test_first, y_test_second), axis = 0)

	length_first_train_set = len(x_train_first)
	length_second_train_set = len(x_train_second)

	length_first_test_set = len(x_test_first)
	length_second_test_set = len(x_test_second)

	permutation_train = np.random.permutation(length_first_train_set + length_second_train_set)
	permutation_test = np.random.permutation(length_first_test_set + length_second_test_set)

	x_train_combined = x_train_combined[permutation_train]
	y_train_combined = y_train_combined[permutation_train]

	x_test_combined = x_test_combined[permutation_test]
	y_test_combined = y_test_combined[permutation_test]

	# Transform the y label to 0 or 1
	y_train_combined = (y_train_combined == target_2)
	y_test_combined = (y_test_combined == target_2)

	# x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
	x_train = torch.from_numpy(x_train_combined).type(torch.FloatTensor)

	# y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_train = torch.from_numpy(y_train_combined).type(torch.LongTensor)

	x_test = torch.from_numpy(x_test_combined).type(torch.FloatTensor)

	# y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_test = torch.from_numpy(y_test_combined).type(torch.LongTensor)

	return x_train, y_train, x_test, y_test


def data_loading_three_target(target_1, target_2, target_3, padding = False, size_for_each_class = 0):
	# Define the first data be labeled 0
	# Define the second data be labeled 1
	x_train_first = None
	y_train_first = None
	x_test_first = None
	y_test_first = None

	x_train_second = None
	y_train_second = None
	x_test_second = None
	y_test_second = None

	x_train_third = None
	y_train_third = None
	x_test_third = None
	y_test_third = None

	if padding == True:
		x_train_first, y_train_first, x_test_first, y_test_first = data_loading_with_padding_target(target_1)
		x_train_second, y_train_second, x_test_second, y_test_second = data_loading_with_padding_target(target_2)
		x_train_third, y_train_third, x_test_third, y_test_third = data_loading_with_padding_target(target_3)

	else:
		x_train_first, y_train_first, x_test_first, y_test_first = data_loading_with_target(target_1)
		x_train_second, y_train_second, x_test_second, y_test_second = data_loading_with_target(target_2)
		x_train_third, y_train_third, x_test_third, y_test_third = data_loading_with_target(target_3)

	if size_for_each_class:
		x_train_first = x_train_first[:size_for_each_class]
		y_train_first = y_train_first[:size_for_each_class]

		x_train_second = x_train_second[:size_for_each_class]
		y_train_second = y_train_second[:size_for_each_class]

		x_train_third = x_train_third[:size_for_each_class]
		y_train_third = y_train_third[:size_for_each_class]

	# Change the target to 0 1 2
	y_train_first = y_train_first - target_1
	y_train_second = y_train_second - target_2 + 1
	y_train_third = y_train_third - target_3 + 2

	y_test_first = y_test_first - target_1
	y_test_second = y_test_second - target_2 + 1
	y_test_third = y_test_third - target_3 + 2

	# Verify the y target.
	# print(y_train_first[:10])
	# print(y_train_second[:10])
	# print(y_train_third[:10])

	# print(y_test_first[:10])
	# print(y_test_second[:10])
	# print(y_test_third[:10])


	x_train_combined = np.concatenate((x_train_first, x_train_second, x_train_third), axis = 0)
	y_train_combined = np.concatenate((y_train_first, y_train_second, y_train_third), axis = 0)

	x_test_combined = np.concatenate((x_test_first, x_test_second, x_test_third), axis = 0)
	y_test_combined = np.concatenate((y_test_first, y_test_second, y_test_third), axis = 0)

	length_first_train_set = len(x_train_first)
	length_second_train_set = len(x_train_second)
	length_third_train_set = len(x_train_third)

	length_first_test_set = len(x_test_first)
	length_second_test_set = len(x_test_second)
	length_third_test_set = len(x_test_third)

	permutation_train = np.random.permutation(length_first_train_set + length_second_train_set + length_third_train_set)
	permutation_test = np.random.permutation(length_first_test_set + length_second_test_set + length_third_test_set)

	x_train_combined = x_train_combined[permutation_train]
	y_train_combined = y_train_combined[permutation_train]

	x_test_combined = x_test_combined[permutation_test]
	y_test_combined = y_test_combined[permutation_test]

	# Transform the y label to 0 or 1
	# Need to resolve when is is ternary
	# y_train_combined = (y_train_combined == target_2)
	# y_test_combined = (y_test_combined == target_2)

	# x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
	x_train = torch.from_numpy(x_train_combined).type(torch.FloatTensor)

	# y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_train = torch.from_numpy(y_train_combined).type(torch.LongTensor)

	x_test = torch.from_numpy(x_test_combined).type(torch.FloatTensor)

	# y_train = torch.from_numpy(y_train).type(torch.LongTensor)
	y_test = torch.from_numpy(y_test_combined).type(torch.LongTensor)

	return x_train, y_train, x_test, y_test

def padding_data(x):
	x = np.pad(x, ((0,0),(2,2),(2,2)), 'constant')
	return x

def load_binary_with_padding(target_num_1 = 0, target_num_2 = 1):

	x_train, y_train, x_test, y_test = data_loading_two_target(target_1 = target_num_1, target_2 = target_num_2, padding = True)

	np.random.seed(0)
	num_data = len(x_train)
	num_train = int(0.75 * num_data)

	index = np.random.permutation(range(num_data))

	x_for_train = x_train[index[:num_train]]
	y_for_train = y_train[index[:num_train]]

	x_for_val = x_train[index[num_train:]]
	y_for_val = y_train[index[num_train:]]

	x_for_test = x_test
	y_for_test = y_test

	return x_for_train, y_for_train, x_for_val, y_for_val, x_for_test, y_for_test

def load_full_class_with_padding():

	x_train, y_train, x_test, y_test = data_loading_with_padding_full()

	np.random.seed(0)
	num_data = len(x_train)
	num_train = int(0.75 * num_data)

	index = np.random.permutation(range(num_data))

	x_for_train = x_train[index[:num_train]]
	y_for_train = y_train[index[:num_train]]

	x_for_val = x_train[index[num_train:]]
	y_for_val = y_train[index[num_train:]]

	x_for_test = x_test
	y_for_test = y_test

	return x_for_train, y_for_train, x_for_val, y_for_val, x_for_test, y_for_test

def main():
	# x_train, y_train, x_test, y_test = data_loading_two_target(0,1,padding = True)
	# x_train, y_train, x_test, y_test = data_loading_one_target_vs_others(target_num = 1)
	# # print("x_train_0: ", x_train_0)
	# # print("x_train_0 shape: ", x_train_0.shape)

	# # print("y_train_0: ", y_train_0)
	# # print("Y_train_0 shape: ", y_train_0.shape)

	# # print("x_train_1: ", x_train_1)
	# # print("x_train_1 shape: ", x_train_1.shape)

	# # print("y_train_1: ", y_train_1)
	# # print("Y_train_1 shape: ", y_train_1.shape)

	# print("x_train: ", x_train)
	# print("x_train.shape: ", x_train.shape)

	# print("MAX: ",x_train.max())
	# print("MIN: ",x_train.min())

	# print("y_train: ", y_train)
	# print("y_train.shape: ", y_train.shape)

	# print("x_test: ", x_test)
	# print("x_test.shape: ", x_test.shape)

	# print("y_test: ", y_test)
	# print("y_test.shape: ", y_test.shape)

	# for i in range(10):
	# 	display_img(x_train[i].numpy().reshape(32,32))

	# x_0, _, _, _ = data_loading_with_padding_target(target_num = 0)

	# x_1, _, _, _ = data_loading_with_padding_target(target_num = 1)

	# x_0 = x_0.numpy()
	# x_1 = x_1.numpy()

	# x_0_square_sum = np.sum(x_0 ** 2, axis = 1).reshape(len(x_0),1)
	# print(x_0_square_sum.shape)
	# # x_0_sqrt = np.clip(np.sqrt(x_0_square_sum), a_min = 1e-9, a_max = None)
	# x_0_sqrt = np.sqrt(x_0_square_sum)
	# print(x_0_sqrt.shape)
	# x_0 = x_0 / x_0_sqrt
	# print(x_0.shape)

	# x_1_square_sum = np.sum(x_1 ** 2, axis = 1).reshape(len(x_1),1)
	# print(x_1_square_sum.shape)
	# # x_1_sqrt = np.clip(np.sqrt(x_1_square_sum), a_min = 1e-9, a_max = None)
	# x_1_sqrt = np.sqrt(x_1_square_sum)
	# print(x_1_sqrt.shape)
	# x_1 = x_1 / x_1_sqrt
	# print(x_1.shape)

	# x_0 = x_0.mean(axis = 0)
	# x_1 = x_1.mean(axis = 0)

	# plot_two_classes(x_0, x_1)
	# 2019 10 28 : MNIST DATA BEHAVE CORRECTLY EVEN WITHOUT CLIP IN THE NORMALIZATION


	x_train, y_train, x_test, y_test = data_loading_three_target(target_1 = 5, target_2 = 7, target_3 = 9, padding = False, size_for_each_class = 0)



	print(x_train)
	print(y_train[:10])

	print(x_test)
	print(y_test[:10])


if __name__ == '__main__':
	main()
