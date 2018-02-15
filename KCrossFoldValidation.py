from Functions import *

model = createModel()
dataset = loadDataset()
data = dataset.data
target = dataset.target
X=PCA(2,data[:,0:4])
X_train,X_test,Y_train,Y_test =SplitTrainingTestData(X,target)
scores1 = KfoldValidation(5,model,X_train,Y_train)
out=one_hot(Y_test)
scores = model.evaluate(X_test,out, verbose=0)
print("Testing on model trained "+" %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save('TrainedModel.model')


