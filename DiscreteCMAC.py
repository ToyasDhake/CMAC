import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from random import randint


def generateAssociationLayer(inputTrain, ansTrain, associtaionWeights, associationBlockSize, numberOfWeights):
    inputs = set(inputTrain)
    for i in inputs:
        i = round(float(i * numberOfWeights), 4)
        for j in range(int(i), int(i) + associationBlockSize + 1):
            associtaionWeights[i][j] = 0
    table = dict()
    count = 0
    for i in inputs:
        i = round(float(i * numberOfWeights), 4)
        table[i] = ansTrain[count]
        count += 1


def train(inputTrain, ansTrain, numberOfWeigths, associtaionWeights, associationBlockSize):
    generation = 1
    while generation <= 300:
        count = 0
        for i in inputTrain:
            mapValue = round(float(i * numberOfWeigths), 4)

            if mapValue not in associtaionWeights.keys():
                continue

            if mapValue in associtaionWeights.keys():
                weight_sum = 0
                for v in associtaionWeights[mapValue].values():
                    weight_sum = weight_sum + v

                error = ansTrain[count] - weight_sum
                changeInError = error / associationBlockSize

                if error >= 0:
                    for idx, k in enumerate(associtaionWeights[mapValue].keys()):
                        if idx == 0:
                            associtaionWeights[mapValue][k] = changeInError
                        elif idx == associationBlockSize:
                            associtaionWeights[mapValue][k] = changeInError
                        else:
                            associtaionWeights[mapValue][k] = changeInError

            count += 1
        generation += 1


def nearestVal(value, associtaionWeights):
    t = [abs(value - x) for x in associtaionWeights.keys()]
    return t.index(min(t))


def test(inputTest, numberOfWeights, associationWeights):
    predictions = list()
    for idx, t in enumerate(inputTest):
        mapValue = round(float(t * numberOfWeights), 4)
        if mapValue not in associationWeights.keys():
            k = nearestVal(mapValue, associationWeights)
            newMapValue = list(associationWeights.keys())[k]
            mapValue = newMapValue

        weightSum = 0
        for l in associationWeights[mapValue].values():
            weightSum = weightSum + l
        predictions.append(weightSum)
    return predictions


data = np.linspace(0, np.pi, 100)
inputTrain = []
for _ in range(70):
    index = randint(0, len(data) - 1)
    inputTrain.append(data[index])
    data = np.delete(data, index)
inputTest = []
for i in data:
    inputTest.append(i)
ansTrain = np.sin(inputTrain)
ansTest = np.sin(inputTest)

associtaionWeights = defaultdict(dict)
numberOfWeights = 35

associationBlockSize = int(input("Association block size-"))

generateAssociationLayer(inputTrain, ansTrain, associtaionWeights, associationBlockSize, numberOfWeights)
train(inputTrain, ansTrain, numberOfWeights, associtaionWeights, associationBlockSize)
predictions = test(inputTest, numberOfWeights, associtaionWeights)

plt.subplot(1, 2, 1)
plt.plot(inputTest, predictions, color="red", linewidth="1.5")
plt.plot(inputTest, ansTest, color="black", linewidth="1.5")
plt.title('Actual Vs Prediction')
plt.legend(['Prediction', 'Actual'])

time = []
for i in range(30, 0, -1):
    time.append(i)
acc = []
for i, _ in enumerate(predictions):
    acc.append(predictions[i] - inputTest[i])
plt.subplot(1, 2, 2)
plt.plot(time, acc, color="blue", linewidth="1.5", linestyle="-")
plt.title('Accuracy over Convergence Time')
plt.legend(['Accuracy', 'Convergence Time'])

plt.show()
