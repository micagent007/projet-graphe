
import numpy as np
import random as rd

####################################################################################

def Binomiale(p):
    if rd.random() < p:
        return 1
    else:
        return 0

def loi_uniforme(K):
    res = np.ones(K)/K
    return res

def W_SSBM(K, a, b):
    res = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            if i == j:
                res[i, j] = a
            else:
                res[i, j] = b
    return res

def tirage(PI):
    s=0
    U=rd.random()
    i=-1
    while U>s:
        s+=PI[i]
        i+=1
    return i

def gen(n,K,W,PI):
    A=np.zeros((n,n))
    X=[tirage(PI) for i in range(n)]
    for i in range(n):
        for j in range(n):
            if (i!=j):
                A[i,j]+=Binomiale(W[X[i],X[j]])
    return A

####################################################################################
        
        
class GraphSSBM:
    def __init__(self, nb_personnes, nb_commu ):
        n = nb_personnes
        K = nb_commu
        Pi = []
        a = 0
        b = 0
        W = np.zeroes((K,K))
        Adj = np.zeroes((n,n))
    
    def afficher(self):
        print(self.Adj)

    def generer_aleatoirement(self, max_a, max_b):
        random_pi = [rd.random() for i in range(0,self.K)] # On génère aléatoirement les L[i]
        normalisation = 0
        for i in range(self.K):
            normalisation += random_pi[i]
        random_pi = [random_pi[i]/normalisation for i in range(0,self.K)] # On normalise pour que "somme des L[i]" = 1
        self.Pi = random_pi
        
        self.a = rd.random()*max_a # On génère a et b aléatoirement
        self.b = rd.random()*max_b
        
        W = W_SSBM(self.K,self.a,self.b) # On génère W à partir des données aléatoires 
        
        Adj = gen(self.n, self.K, self.W, self.Pi) # On génère la matrice d'adjacence à partir des a et b aléatoires et de random_pi
        
    def generer(self, val_a, val_b, pi):
        self.Pi = pi
        self.a = val_a
        self.b = val_b
        
        W = W_SSBM(self.K,self.a,self.b)
        
        Adj = gen(self.n, self.K, self.W, self.Pi)
        
    def acces_adj(self):
        return(self.Adj)
        
        
            
        
    

####################################################################################




        
    