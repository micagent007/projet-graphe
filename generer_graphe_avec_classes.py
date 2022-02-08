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
        for j in range(i, K):
            r = rd.random()
            res[i][j] = r
            res[j][i] = r
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
    return ([k for k in range(1, n+1)])


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
    
    def trac_graph_communaute(self,com):
        G = nx.Graph()
        G.add_nodes_from(liste(self.n))
        A = self.Adj
        for i in range(self.n):
            for j in range(self.n):
                if A[i][j] == 1:
                    G.add_edge(i + 1, j + 1)
                    
        values=[0 for k in range(self.n)]             
        for k in range(len(com)):
            for j  in range (len(com[k])):
                values[com[k][j]-1]=k
        options = {
        'node_color' : values,
        'node_size'  : 500,
        'edge_color' : 'tab:grey',
        'with_labels': True
        }
        nx.draw(G,**options)
        plt.savefig("test.png")    


####################################################################################

# tests

"""G = GraphSBM(30, 5)
G.generer_aleatoirement_SSBM(1,0)
G.afficher()
G.trac_graph()"""


####################################################################################

# K-means

def norme(v):
    n = len(v)
    somme = 0
    for k in range(n):
        somme += v[k] ** 2
    return np.sqrt(somme)


def plus_proche(dep, v):
    dep = dep.transpose()
    mini, ind = norme(v - dep[0]), 0
    for k in range(len(dep)):
        c=norme(v - dep[k])
        if (mini > c ):
            ind = k
            mini = norme(v - dep[k])
    dep = dep.transpose()
    return (ind)


def barycentre(vect):  # vect array de p vecteur à n coordonnées
    bary = []
    for i in range(len(vect)):
        somme = 0
        for j in range(len(vect[0])):
            somme += vect[i][j]
        bary += [somme / len(vect)]
    return bary


def laplace(adj):
    D = np.diag(adj.sum(axis=0))
    return D - adj


def vp_laplacien(adj):
    (x, y) = np.linalg.eig(laplace(adj))
    return (x, y)


def spectral_clustering(adj):
    (Vap, Vep) = vp_laplacien(adj)
    K, n, Nbvect = 0, len(Vep), len(Vap)
    Vep = Vep.transpose()
    L = []
    for k in range(Nbvect):
        if abs(Vap[k]) < 10 ** (-10):
            K += 1
            L += [Vep[k]]
    vect = np.ones((K, n))
    for i in range(n):
        for j in range(K):
            vect[j][i] = L[j][i]
    return ((K, vect))


def K_means(K, vect):  # vect liste de vecteurs propres du laplacien
    n=len(vect)
    nbrevect = len(vect[0])
    vect2 = vect.copy()
    vect2 = vect2.transpose()
    VECT=[vect2[k] for k in  range(nbrevect)]
    DEP = []
    for k in range(K):
        DEP += [VECT.pop(rd.randint(0, len(VECT)-1))]
    dep = np.ones((n, K))
    for i in range(n):
        for j in range(K):
            dep[i][j] = DEP[j][i]
    L = []
    vect = vect.transpose()
    for j in range(nbrevect):
        L += [plus_proche(dep, vect[j])]
    vect=vect.transpose()
    Lancien=L
    stop= False
    while (stop==False):
        stop=True
        PCOM=[]
        Lnouveau=[]
        Com=[[] for k in range(K)]
        for k in range(len(L)):
            Com[L[k]]+=[k+1]
        for l in range(K):
            nbrpoint=len(Com[l])
            M=np.ones((n,nbrpoint))
            for i in range(n):
                for j in range(nbrpoint):
                    M[i][j]=vect[i][Com[l][j]-1]
            PCOM+=[barycentre(M)]
        vect = vect.transpose()
        pcom = np.ones((n, K))
        for i in range(n):
            for j in range(K):
                pcom[i][j] = PCOM[j][i]
        for m in range(nbrevect):
            Lnouveau += [plus_proche(pcom, vect[m])]
            if (Lnouveau[-1]!=L[m]):
                stop=False
        vect = vect.transpose()
        L=Lnouveau
    return(Com)










####################################################################################
G = GraphSBM(20, 2)
G.generer_aleatoirement_SSBM(1, 0)
G.afficher()

(K, vect) = spectral_clustering(G.Adj)
print (K)
com=K_means(K, vect)
print(len(com))
print(com)
G.trac_graph_communaute(com)

