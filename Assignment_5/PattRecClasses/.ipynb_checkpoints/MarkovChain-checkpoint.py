import numpy as np
from .DiscreteD import DiscreteD

class MarkovChain:
    """
    MarkovChain - class for first-order discrete Markov chain,
    representing discrete random sequence of integer "state" numbers.
    
    A Markov state sequence S(t), t=1..T
    is determined by fixed initial probabilities P[S(1)=j], and
    fixed transition probabilities P[S(t) | S(t-1)]
    
    A Markov chain with FINITE duration has a special END state,
    coded as nStates+1.
    The sequence generation stops at S(T), if S(T+1)=(nStates+1)
    """
    def __init__(self, initial_prob, transition_prob):

        self.q = initial_prob  #InitialProb(i)= P[S(1) = i]
        self.A = transition_prob #TransitionProb(i,j)= P[S(t)=j | S(t-1)=i]


        self.nStates = transition_prob.shape[1] # number of cloumns i.e states

        self.is_finite = False
        if self.A.shape[0] != self.A.shape[1]: # is finite if number of columns does not equal number of rows. 
            self.is_finite = True
            self.end_state = self.nStates - 1 # index of final state fixed for now as last index
  

    def probDuration(self, tmax):
        """
        Probability mass of durations t=1...tMax, for a Markov Chain.
        Meaningful result only for finite-duration Markov Chain,
        as pD(:)== 0 for infinite-duration Markov Chain.
        
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.8.
        """
        pD = np.zeros(tmax)

        if self.is_finite:
            pSt = (np.eye(self.nStates)-self.A.T)@self.q

            for t in range(tmax):
                pD[t] = np.sum(pSt)
                pSt = self.A.T@pSt

        return pD

    def probStateDuration(self, tmax):
        """
        Probability mass of state durations P[D=t], for t=1...tMax
        Ref: Arne Leijon (201x) Pattern Recognition, KTH-SIP, Problem 4.7.
        """
        t = np.arange(tmax).reshape(1, -1)
        aii = np.diag(self.A).reshape(-1, 1)
        
        logpD = np.log(aii)*t+ np.log(1-aii)
        pD = np.exp(logpD)

        return pD

    def meanStateDuration(self):
        """
        Expected value of number of time samples spent in each state
        """
        return 1/(1-np.diag(self.A))
    
    def rand(self, tmax):
        """
        S=rand(self, tmax) returns a random state sequence from given MarkovChain object.
        
        Input:
        tmax= scalar defining maximum length of desired state sequence.
           An infinite-duration MarkovChain always generates sequence of length=tmax
           A finite-duration MarkovChain may return shorter sequence,
           if END state was reached before tmax samples.
        
        Result:
        S= integer row vector with random state sequence,
           NOT INCLUDING the END state,
           even if encountered within tmax samples
        If mc has INFINITE duration,
           length(S) == tmax
        If mc has FINITE duration,
           length(S) <= tmaxs
        """
        
        S = np.empty(tmax, dtype=int);
        print(self.nStates)
        print(self.q)
        S[0] = np.random.choice(self.nStates, p=self.q);

        for i in range(1, tmax):
            prev_state = S[i-1]
            pi = self.A[prev_state]
            S[i] = np.random.choice(self.nStates, p=pi);
            if self.is_finite and S[i] == self.end_state:
                return S[:i+1]
        return S

    def viterbi(self):
        pass
    
    def stationaryProb(self):
        pass
    
    def stateEntropyRate(self):
        pass
    
    def setStationary(self):
        pass

    def logprob(self):
        pass

    def join(self):
        pass

    def initLeftRight(self):
        pass
    
    def initErgodic(self):
        pass

    def forward(self, scaledProbOfObservations):
        T = scaledProbOfObservations.shape[1]
        J = self.A.shape[0]
        alpha = np.zeros((J, T))
        c = np.zeros(T)
        temp = np.zeros(J)

        #Initialization step
        temp[0] = self.q.dot(scaledProbOfObservations[:, 0])
        c[0] = np.sum(temp)
        alpha[:, 0] = temp / c[0]
        
        #Forward step
        for t in range(1, T):
            for j in range(J):
                temp[j] = alpha[:, t - 1].dot(self.A[:, j]) * scaledProbOfObservations[j, t]
            c[t] = np.sum(temp)
            alpha[:,t] = temp/c[t]

        #Termination step for finite chains
        if self.is_finite:
            c = np.append(c, [alpha[:, alpha.shape[1] - 1].dot(self.A[:, self.A.shape[1] - 1])])
            
        return alpha, c
    

    def finiteDuration(self):
        pass
    
    def backward(self, scaledProbOfObservations, c):
        T = scaledProbOfObservations.shape[1]
        J = self.A.shape[0]
        beta = np.zeros((J, T))
        one = np.ones(J)
        
        #Initialization Step
        if self.is_finite:
            beta[:, T - 1] = self.A[:, J] / (c[T - 1] * c[T])
        else:
            beta[:, T - 1] = one / c[T - 1]


        #Backward Step
        for t in range(T - 2, -1, -1): #Starting with T-1 at index T - 2 
            for i in range(J):
                probThatiCameBeforej = 0
                for j in range(J):
                    probThatiCameBeforej += self.A[i, j] * beta[j, t + 1] * scaledProbOfObservations[j, t + 1]
                beta[i, t] += probThatiCameBeforej
            beta[:, t] = beta[:, t] / c[t]

            
        return beta

    def adaptStart(self):
        pass

    def adaptSet(self):
        pass

    def adaptAccum(self):
        pass
