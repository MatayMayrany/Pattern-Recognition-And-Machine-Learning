import numpy as np
from PattRecClasses import GaussD, HMM, MarkovChain


class IsolatedWordRecognizer:

    #used https://tophonetics.com/
    mapOfWordToPhenomeCount = {
        'yes': 3,
        'no': 2,
        'up': 2,
        'down': 4,
        'go': 3,
        'stop': 4,
        'on': 2,
        'off': 2,
        'right': 4,
        'left': 4
    }
    
    def __init__(self, word):
        self.word = word
        self.phonemesNumber = self.mapOfWordToPhenomeCount[word]
        self.HMM = None
        
    def saveModel(self):
        np.save("models/" + self.word + "/Q", self.HMM.stateGen.q)
        np.save("models/" + self.word + "/A", self.HMM.stateGen.A)
        
        means = []
        covs = []
        for distribution in self.HMM.outputDistr:
            means += [distribution.means]
            covs += [distribution.cov]

        np.save("models/" + self.word + "/B_means", means)
        np.save("models/" + self.word + "/B_covs", covs)
        
    def loadModel(self):
        Q = np.load("models/" + self.word + "/Q.npy")
        A = np.load("models/" + self.word + "/A.npy")
        means = np.load("models/" + self.word + "/B_means.npy")
        covs = np.load("models/" + self.word + "/B_covs.npy")
        
        mc = MarkovChain(Q, A) 
        B = [GaussD( means[i], cov=covs[i] ) for i in range(len(means))] 
        
        self.HMM = HMM(mc, B)
        
    def getInitialModel(self, averageNumberOfFeatures, featureSize):
        stateNumber = self.phonemesNumber + 2 # silence and exit

        # Initial distributuion even for all states
        initalProbForAnyState = 1 / (stateNumber - 1)
        Q = np.repeat([initalProbForAnyState], stateNumber)
        Q[stateNumber - 1] = 0 # dont start at exit
        print(Q)

        # initial Transition Matrix
        A = np.zeros((stateNumber - 1, stateNumber)) #finite
        for i in range(stateNumber - 1):
            for j in range(stateNumber - 1):
                A[i, j] = 1 / (stateNumber - 1)                
        halfProbOfExit = A[stateNumber - 2][stateNumber - 2] / 2
        A[stateNumber - 2][stateNumber - 2] = halfProbOfExit
        A[stateNumber - 2][stateNumber - 1] = halfProbOfExit
        print(A)
        
        # initial Emission Matrix
        B = [GaussD(np.repeat([0], featureSize), stdevs=2) for i in range(stateNumber)]
        print(B)


        initialMarkovChain = MarkovChain(Q, A)
        initialHMM = HMM(initialMarkovChain, B)
        return initialHMM
    
    def train(self, dataset): 
        averageNumberOfFeatures = int(np.mean([datapoint.shape[1] for datapoint in dataset])) 
        self.HMM = self.getInitialModel(averageNumberOfFeatures, dataset[0].shape[0])
        return self.HMM.baum_welch(dataset, 2) #10 iter
        
    def loadModel(self):
        Q = np.load("models/" + self.word + "/Q.npy")
        A = np.load("models/" + self.word + "/A.npy")
        means = np.load("models/" + self.word + "/B_means.npy")
        covs = np.load("models/" + self.word + "/B_covs.npy")
        
        mc = MarkovChain(Q, A) 
        B = [GaussD( means[i], cov=covs[i] ) for i in range(len(means))] 
        
        self.HMM = HMM(mc, B)

    def evaluateModel(self, dataset):
        scores = []
        for datapoint in dataset:
            scores.append(self.HMM.logprob(datapoint))
        return scores
    
    def evaluateDataPoint(self, datapoint):
        return self.HMM.logprob(datapoint)
           