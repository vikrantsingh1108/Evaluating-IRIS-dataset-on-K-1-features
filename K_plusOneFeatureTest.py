from Functions import *

plt.figure(1)
model = createModel()
dataset = loadDataset()
data = dataset.data
target = dataset.target
X=PCA(2,data[:,0:2])
X_train,X_test,Y_train,Y_test =SplitTrainingTestData(X,target)
scores = KfoldValidation(5,model,X_train,Y_train)
out=one_hot(Y_test)
scores1  = model.evaluate(X_test,out, verbose=0)
print("Testing on model trained "+" %s: %.2f%%" % (model.metrics_names[1], scores1[1]*100))
model.save('TrainedModel.model')


model = load_model("TrainedModel.model")
dataset = loadDataset()
data = dataset.data
target = dataset.target
X=PCA(2,data[:,0:3])
out = one_hot(target)
scores2 = model.evaluate(X, out, verbose=0)
print("Accuracy after introducing petal length:" +"%s: %.2f%%" % (model.metrics_names[1], scores2[1]*100))

dataset = loadDataset()
data = dataset.data
target = dataset.target
X=PCA(2,data[:,0:4])
out = one_hot(target)
scores3 = model.evaluate(X, out, verbose=0)
print("Accuracy after introducing petal width:"+"%s: %.2f%%" % (model.metrics_names[1], scores3[1]*100))


avg_scores =[]
avg_scores.append(scores1[1]*100)
avg_scores.append(scores2[1]*100)
avg_scores.append(scores3[1]*100)

#print avg_scores
plt.subplot(211)
plotAccuracyCurve(scores,'model accuracy with 2 features',0,'Validation steps')

plt.subplot(212)
plotAccuracyCurve(avg_scores , "Overall Accuracy Graph" , 1 ,'no. of features')

plt.show()

