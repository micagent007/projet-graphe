import numpy as np
import random as rd
import networkx as nx
import matplotlib.pyplot as plt


####################################################################################

def Binomiale(p):
    if rd.random() < p:
        return 1
    else:
        return 0


def loi_uniforme(K):
    res = np.ones(K) / K
    return res


def W_SSBM(K, a, b):
    res = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                res[i, j] = a
            else:
                res[i, j] = b
    return res


def W_SBM(K):
    res = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            res[i][j] = rd.random()
    return res


def proba_quelconque(K):
    random_pi = [rd.random() for i in range(K)]  # On génère aléatoirement les L[i]
    normalisation = 0
    for i in range(K):
        normalisation += random_pi[i]
    random_pi = [random_pi[i] / normalisation for i in range(0, K)]  # On normalise pour que "somme des L[i]" = 1
    return random_pi


def tirage(PI):
    s = 0
    U = rd.random()
    i = -1
    while U > s:
        s += PI[i]
        i += 1
    return i


def liste(n):
    return ([k for k in range(1, n + 1)])


def gen(n, K, W, PI):
    A = np.zeros((n, n))
    X = [tirage(PI) for i in range(n)]
    for i in range(n):
        for j in range(n):
            if (i < j):
                A[i, j] += Binomiale(W[X[i], X[j]])
    A = A + np.transpose(A)
    return A


####################################################################################


class GraphSBM:
    def __init__(self, nb_personnes, nb_commu):
        self.n = nb_personnes
        self.K = nb_commu
        self.Pi = []
        self.a = 0
        self.b = 0
        self.W = np.zeros((self.K, self.K))
        self.Adj = np.zeros((self.n, self.n))

    def afficher(self):
        print(self.Adj)

    def generer_aleatoirement_SBM(self):
        self.Pi = proba_quelconque(self.K)

        self.W = W_SBM(
            self.K)  # On génère W aléatoirement (matrice des coeffs de bernoulli pour la matrice d'adjacence)

        self.Adj = gen(self.n, self.K, self.W,
                       self.Pi)  # On génère la matrice d'adjacence à partir des a et b aléatoires et de random_pi

    def generer_aleatoirement_SSBM(self, max_a, max_b):
        self.Pi = loi_uniforme(self.K)

        self.a = rd.random() * max_a  # On génère a et b aléatoirement
        self.b = rd.random() * max_b

        self.W = W_SSBM(self.K, self.a, self.b)  # On génère W à partir des données aléatoires

        self.Adj = gen(self.n, self.K, self.W,
                       self.Pi)  # On génère la matrice d'adjacence à partir des a et b aléatoires et de random_pi

    def generer_SSBM(self, val_a, val_b):
        self.Pi = loi_uniforme(self.K)
        self.a = val_a
        self.b = val_b

        self.W = W_SSBM(self.K, self.a, self.b)

        self.Adj = gen(self.n, self.K, self.W, self.Pi)

    def generer_predefini(self, A):
        self.adj = A
        self.n = len(A)

    def acces_adj(self):
        return (self.Adj)

    def trac_graph(self):
        G = nx.Graph()
        G.add_nodes_from(liste(self.n))
        A = self.Adj
        for i in range(self.n):
            for j in range(self.n):
                if A[i][j] == 1:
                    G.add_edge(i + 1, j + 1)
        nx.draw(G)
        plt.savefig("test.png")


####################################################################################


G = GraphSBM(30, 2)
G.generer_aleatoirement_SBM()
G.afficher()
G.trac_graph()