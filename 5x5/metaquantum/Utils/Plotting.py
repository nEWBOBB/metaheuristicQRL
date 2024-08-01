'''
2019 11 16
Add some code, need to edit.

'''

import matplotlib.pyplot as plt
from datetime import datetime

def plotTrainingResult(_iterationIndex, _costTrainList, _costTestList, _accuracyTraining, _accuracyValidation,  _accuracyTest, _exp_name, _fileTitle):
	fig, ax = plt.subplots()
	# plt.yscale('log')
	ax.plot(_iterationIndex, _accuracyTraining, '-b', label='Accuracy Training')
	ax.plot(_iterationIndex, _accuracyValidation, '-r', label='Accuracy Validation')
	ax.plot(_iterationIndex, _accuracyTest, '-c', label='Accuracy Test')
	ax.plot(_iterationIndex, _costTrainList, '-g', label='Cost Train')
	ax.plot(_iterationIndex, _costTestList, '-m', label='Cost Test')
	leg = ax.legend();

	ax.set(xlabel='Iteration', 
		   title=_exp_name)
	fig.savefig(_fileTitle + "_"+ datetime.now().strftime("NO%Y%m%d%H%M%S") + ".pdf", format='pdf')