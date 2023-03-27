import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

def costfunction(trainDataset, trainTarget, particle, features):
    particle = particle.tolist()
    #particle = particle[0]
    redundantFeatures = []
    for j in range(len(particle)):
        if (particle[j] == 0):
            redundantFeatures.append(j)
    for i in redundantFeatures:
        trainDataset = trainDataset.drop(columns=features[i])
    clf = svm.SVC(kernel='rbf', C=1, gamma=1)  # rbf Kernel
    clf.fit(trainDataset, trainTarget)
    y_pred = clf.predict(trainDataset)
    accuracy = accuracy_score(trainTarget, y_pred)
    error = 1-accuracy
    return error


def initialize(popSize, dimension):
    pop = []
    for i in range(popSize):
        particle = np.random.rand(1, dimension)
        particle[particle > 0.5] = 1
        particle[particle <= 0.5] = 0
        pop.append(particle[0])
    return pop

def train(dataset, target, popSize, MaxIt):
    dimension = dataset.shape[1]
    features = dataset.columns.values.tolist()
    cost = []
    pbestCost = []
    w = 1
    c1 = 2
    c2 = 2
    wdamp = 0.99
    globalCost = float('inf')
    globaCost_it = []
    averageCost_it = []
    pbestPos = []
    particleVelocity = []
    ## initialization
    population = initialize(popSize, dimension)
    for i in range(popSize):
        error = costfunction(dataset, target, population[i], features)
        cost.append(error)
        particleVelocity.append(np.zeros(dimension))
        pbestPos.append(population[i])
        pbestCost.append(error)
        if (error < globalCost):
            globalCost = error
            globalPos = population[i]


    ### main loop

    for it in range(MaxIt):
        for i in range(popSize):
            particle = population[i]
            particleVelocity[i] = w*particleVelocity[i] + c1*np.random.rand()*(pbestPos[i] - particle) +\
                                  c2*np.random.rand()*(globalPos - particle)
            particle = particle + particleVelocity[i]
            particle[particle > 0.5] = 1
            particle[particle <= 0.5] = 0
            population[i] = particle
            error = costfunction(dataset, target, population[i], features)
            cost[i] = error
            if (error < globalCost):
                globalCost = error
                globalPos = population[i]

            if (error < pbestCost[i]):
                pbestCost[i] = error
                pbestPos[i] = population[i]

        w = w * wdamp






