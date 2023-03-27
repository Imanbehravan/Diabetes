from parallelAPSO.PSOClustering import PSO_Clustering
import pandas as pd
import numpy as np
import math
from sklearn.metrics import rand_score
from parallelAPSO.PSOClustering import assignLabels
import joblib


def runSequence(data, popNum, MaxIt, sequenceSteps, numRepSamples, i):
    print("----------------------sequence number ", i+1, " is started ----------------------")
    print(" ")
    for j in range(sequenceSteps):
        bestK_list = []
        bestCost_list = []
        if (j == 0):
            k = np.random.randint(2, math.sqrt(len(data)))
            bestKSofar = k
            bestK_list.append(bestKSofar)
            cost, pos = PSO_Clustering(data, k, popNum, MaxIt, False, numRepSamples, 1, i, j)
            bestCostSofar = cost
            bestCost_list.append(bestCostSofar)
        else:
            k = k + np.random.randint(-3, 3)
            if (k < 2):
                k = 2
            cost, pos = PSO_Clustering(data, k, popNum, MaxIt, False, numRepSamples, 1, i, j)
            if (cost < bestCostSofar):
                bestCostSofar = cost
                bestKSofar = k
            bestK_list.append(bestKSofar)
            bestCost_list.append(bestCostSofar)
    bestcost_index = bestCost_list.index(min(bestCost_list))
    print("----------------------sequence number ", i+1, " is finished ----------------------")
    print(" ")
    return bestK_list[bestcost_index], min(bestCost_list)


def APSO_Clustering(config, dataset, targets):
    sequenceNum = config["DEFAULT"]["sequenceNum"]
    sequenceSteps = config["DEFAULT"]["sequenceSteps"]
    sequence_PSOPop = config["DEFAULT"]["sequence_PSOPop"]
    sequence_MaxIt = config["DEFAULT"]["sequence_MaxIt"]
    secondStage_PSOpop = config["DEFAULT"]["secondStage_PSOpop"]
    secondStage_MaxIt = config["DEFAULT"]["secondStage_MaxIt"]
    numRepSamples = config["DEFAULT"]["numRepSamples"]
    rand_index_flag = config["DEFAULT"]["rand_index_flag"]
    centroids = []
    outputfile = config["DEFAULT"]["outputpath"]
    #DataPath = config["DEFAULT"]["DataPath"]
    #dataset = pd.read_csv('DataPath')
    #targets = config["DEFAULT"]["targetField"]
    if rand_index_flag:
        target = dataset["targets"]
    #trainDataset = dataset.drop(columns=["targets"])
    result = joblib.Parallel(n_jobs=sequenceNum)(joblib.delayed(runSequence)(dataset, sequence_PSOPop, sequence_MaxIt, sequenceSteps, numRepSamples,i) for i in range(sequenceNum))
    bestCostList = []
    for j in range(sequenceNum):
        bestCostList.append(result[j][1])
    minimumCost = min(bestCostList)
    bestK_index = bestCostList.index(minimumCost)
    bestK = result[bestK_index][0]
    finalBestCost, finalCentroids = PSO_Clustering(dataset, bestK, secondStage_PSOpop, secondStage_MaxIt, False, numRepSamples, 2, 0, 0)
    predLabels = assignLabels(dataset, finalCentroids)
    index = 1
    for i in range(bestK):
        if predLabels.count(i) > 0:
            centroids.append(finalCentroids[i])
            print("number of elements in cluster ", index, "is: ", predLabels.count(i))
            index = index + 1
    print("best number of clusters: ", len(centroids))
    print("best centroids: ", centroids)
    if rand_index_flag:
        rand_index = rand_score(target, predLabels)
        print("rand index: ", rand_index)

    f = open(outputfile, "w")
    f.write("number of centroids: ")
    f.write(str(len(centroids)))
    f.write('\n')
    f.write('\n')
    f.write("centroids: ")
    f.write('\n')
    np.savetxt(f, centroids)
    #f.write(finalCentroids)
    f.write('\n')
    f.write("cost: ")
    #np.savetxt(f,finalBestCost)
    f.write(str(finalBestCost))
    f.write('\n')
    f.write("Rand index: ")
    if rand_index_flag:
        f.write(str(rand_index))

    else:
        f.write("not calculated")

    f.close()
    return centroids



