# Proof of concept code, needs cleaning up with a few more iterations and improvements

import numpy as np
import matplotlib.pyplot as plt

trainData = np.genfromtxt('train.data', delimiter=",")
testData = np.genfromtxt('test.data', delimiter=",")

class Perceptron():

    def __init__(self, learnRate, maxEpoch, bias):
        self.learnRate = learnRate
        self.maxEpoch = maxEpoch
        self.bias = bias

    def train(self, data):
        self.bias = np.full((1),self.bias)
        self.bias = self.bias.astype('float64')
        self.weights = np.zeros((len(data[0]))-1)
        np.random.shuffle(data)
        labels = data[:,4:]
        labels = labels.astype('float64')   
        data = data[:,:4]
        self.errors_ = []
        for epoch in range(self.maxEpoch): 
            errors = 0
            for rowObject, label in zip(data, labels):
                activation = np.dot(self.weights, rowObject) + self.bias
                prediction = (label*activation)
                if prediction <= 0:
                    for i in range(len(self.weights)):
                        self.weights[i] = self.weights[i] + label*rowObject[i]
                    self.bias += label
                    errors += 1     
            self.errors_.append(errors)
        return self.weights, self.bias

    def test(self, data, weights, bias):
        labels = data[:,4:]
        labels = labels.astype('float64')
        data = data[:,:4]
        predictionClass0 = [0,0] # store the classified scores true = [0], false = [1] for each class
        predictionClass1 = [0,0]
        for rowObject, label in zip(data, labels):
            activation = np.dot(self.weights, rowObject) + self.bias
            prediction = (label*activation)
            if prediction <= 0:
                if label == 1: #wrong
                    predictionClass0[1] += 1
                else:
                    predictionClass1[1] += 1
            else: #correct
                if label == 1:
                    predictionClass0[0] += 1
                else:
                    predictionClass1[0] += 1
        print(f'Accuracy: {(predictionClass0[0] + predictionClass1[0])/(len(labels))*100} Precision: {(predictionClass0[0]/(predictionClass0[0]+predictionClass0[1]))*100} Recall: {predictionClass0[0]/sum(predictionClass0)*100} F-Score: {(2*(((predictionClass0[0]/(predictionClass0[0]+predictionClass0[1]))*100)*(predictionClass0[0]/sum(predictionClass0)*100))/(((predictionClass0[0]/(predictionClass0[0]+predictionClass0[1]))*100)+(predictionClass0[0]/sum(predictionClass0)*100)))}')
        print(f'Correct: {predictionClass0[0]+predictionClass1[0]} Wrong: {predictionClass0[1]+predictionClass1[1]}\n')
        return activation

def classPicker(classX, classY):
    set1 = trainData[:40,:4]
    set2 = trainData[40:80,:4]
    set3 = trainData[80:120,:4]
    class0 = np.full((40,1),1)
    class1 = np.full((40,1),-1)
    if classX == 1 and classY == 2:
        return np.vstack([np.hstack([set1,class0]),np.hstack([set2,class1])])
    if classX == 2 and classY == 3:
        return np.vstack([np.hstack([set2,class0]),np.hstack([set3,class1])])
    if classX == 1 and classY == 3:
        return np.vstack([np.hstack([set1,class0]),np.hstack([set3,class1])])

def testPicker(classX, classY):
    set1 = testData[:10,:4]
    set2 = testData[10:20,:4]
    set3 = testData[20:30,:4]
    class0 = np.full((10,1),1)
    class1 = np.full((10,1),-1)
    if classX == 1 and classY == 2:
        return np.vstack([np.hstack([set1,class0]),np.hstack([set2,class1])])
    if classX == 2 and classY == 3:
        return np.vstack([np.hstack([set2,class0]),np.hstack([set3,class1])])
    if classX == 1 and classY == 3:
        return np.vstack([np.hstack([set1,class0]),np.hstack([set3,class1])])

# Compare setosa & versicolor
trainDataset = (classPicker(1,2))
pececptron1 = Perceptron(0,20,0)
pececptron1.train(trainDataset)
plt.plot(range(1, len(pececptron1.errors_) + 1), pececptron1.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()
testDataset = (testPicker(1,2))
pececptron1.test(trainDataset, pececptron1.weights, pececptron1.bias)
pececptron1.test(testDataset, pececptron1.weights, pececptron1.bias)

# Compare versicolor & virginica
trainDataset = (classPicker(2,3))
pececptron2 = Perceptron(0,20,0)
pececptron2.train(trainDataset)
plt.plot(range(1, len(pececptron2.errors_) + 1), pececptron2.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()
testDataset = (testPicker(2,3))
pececptron2.test(trainDataset, pececptron2.weights, pececptron2.bias)
pececptron2.test(testDataset, pececptron2.weights, pececptron2.bias)

# Compare setosa & virginica
trainDataset = (classPicker(1,3))
pececptron3 = Perceptron(0,20,0)
pececptron3.train(trainDataset)
plt.plot(range(1, len(pececptron3.errors_) + 1), pececptron3.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()
testDataset = (testPicker(1,3))
pececptron3.test(trainDataset, pececptron3.weights, pececptron3.bias)
pececptron3.test(testDataset, pececptron3.weights, pececptron3.bias)