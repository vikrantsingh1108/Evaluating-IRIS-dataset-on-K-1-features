from keras.models import Sequential
from keras.layers import Dense ,Dropout
from keras.utils import *
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn.model_selection import train_test_split
import numpy
import seaborn as sns
import pandas as pd

"""
function to plot performace graphs
"""
def plotAccuracyCurve(scores, des , off , x_label):
	plt.title(des)
	plt.grid(True)
	x_plot_data=numpy.linspace(1,len(scores), len(scores))
	x_plot_data+=off
	plt.plot(x_plot_data,scores)
	#plt.annotate('mean/average accuraccy', xy=(3.5, numpy.mean(scores)), xytext=(3.5, 99),arrowprops=dict(facecolor='black', shrink=0.05),)
	plt.axhline(numpy.mean(scores), color='black', lw=2)
	plt.ylabel('Accuracy')
	plt.xlabel(x_label)
	

"""
function to perform Kfold Cross Validation
"""
def KfoldValidation(folds,model,X,Y):
	seed = 7
	numpy.random.seed(seed)
	cvscores = []
	kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
	i=1
	for train, test in kfold.split(X, Y):
		print(train.shape, test.shape)
		# Fit the model
		out=one_hot(Y[train])
		model.fit(X[train], out, nb_epoch=100, batch_size=5, verbose=0)
		# evaluate the model
		out=one_hot(Y[test])
		scores = model.evaluate(X[test], out, verbose=0)
		print("Validation step " +str(i)+" of K fold Validation  "+"%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		cvscores.append(scores[1] * 100)
		i+=1
		
	print("Overall Accuracy after k fold validation "+"%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
	return cvscores

"""
function for one hot encoding using categorical
"""
def one_hot(input_array):
	unique,index=numpy.unique(input_array,return_inverse=True)
	return np_utils.to_categorical(index,len(unique))

"""
function to load iris dataset

"""
def loadDataset():
	dataset = datasets.load_iris()
	return dataset


"""
function to perform PCA(Principal Component Analysis)
"""
def PCA(dim,X):
	pca = decomposition.PCA(n_components=dim)
	pca.fit(X)
	X = pca.transform(X)
	return X

"""
function to split test training data
"""	
def SplitTrainingTestData(X,Y):
	X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=0)
	return X_train, X_test, y_train, y_test


"""
function to createModel
"""
def createModel():
	# create model
	model = Sequential()
	model.add(Dense(3, input_dim=2, init='uniform', activation='relu'))
	model.add(Dense(24, init='uniform', activation='softplus'))
	model.add(Dense(24, init='uniform', activation='tanh'))
	model.add(Dense(3, init='uniform', activation='softmax'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

