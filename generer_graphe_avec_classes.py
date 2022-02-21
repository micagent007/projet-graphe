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


def W_SSBM(K, a, b,alpha_n):
    res = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                res[i, j] = a*alpha_n
            else:
                res[i, j] = b*alpha_n
    return res


def W_SBM(K):
    res = np.zeros((K, K))
    for i in range(K):
        for j in range(i, K):
            r = rd.random()
            res[i][j] = r
            res[j][i] = r
    return res

def W_SBM1(K):
    res = np.zeros((K, K))
    D= np.diag([rd.random() for j in range(K)])
    for i in range(K):
        for j in range(i, K):
            if (i<j):
                res[i,j] = rd.random()
    res+= np.transpose(res)+D
    return res
print (W_SBM1(3))

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

    def generer_aleatoirement_SSBM(self, max_a, max_b, alpha_n):
        self.Pi = loi_uniforme(self.K)

        self.a = rd.random() * max_a  # On génère a et b aléatoirement
        self.b = rd.random() * max_b

        self.W = W_SSBM(self.K, self.a, self.b,alpha_n)  # On génère W à partir des données aléatoires

        self.Adj = gen(self.n, self.K, self.W,
                       self.Pi)  # On génère la matrice d'adjacence à partir des a et b aléatoires et de random_pi

    def generer_SSBM(self, val_a, val_b,alpha_n):
        self.Pi = loi_uniforme(self.K)
        self.a = val_a
        self.b = val_b

        self.W = W_SSBM(self.K, self.a, self.b,alpha_n)

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
                if A[i,j] == 1:
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
        plt.plot()
        plt.savefig("test.png")   
        plt.show()

    def histogramme(self):
        (Vap, Vep) = Vap_Vep(self.Adj)
        idx = np.flip(Vap.argsort()[::-1])
        Vap = Vap[idx]
        Vep = Vep[:, idx]  # Vep et Vap sont ici triées
        X=list(range(self.n//4))
        Y=[Vap[x] for x in X]
        plt.scatter(X,Y,s=20,color='r')
        plt.show()
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


def Vap_Vep(adj):
    (x, y) = np.linalg.eig(laplace(adj))
    x=x.real                                #On prend la partie réelle pour ignorer la partie imaginaire apparaît à cause des approximations de l'ordinateur
    y=y.real
    idx = np.flip(x.argsort()[::-1])
    x = x[idx]
    y = y[:, idx]  # Vep et Vap sont ici trié
    return (x, y)

def spectral_clustering_sans_k(adj):
    (Vap, Vep) = Vap_Vep(adj)
    K, Nbvect = 0, len(Vap)
    Vep = Vep.transpose()
    L = []
    ecartrel=np.array([(abs(Vap[j]-Vap[j+1]))/Vap[j] for j in range(1,Nbvect-1)])
    print(Vap)
    print(ecartrel)
    K=np.argmax(ecartrel)+2
    for k in range(0,K):
        L.append(Vep[k])
    L=np.array(L)
    vect=L.transpose()
    return (K, vect)
def spectral_clustering_avec_k(adj,k):
    (Vap, Vep) = Vap_Vep(adj)
    K = k
    Vep = Vep.transpose()
    L = []
    for j in range(K):
        L.append(Vep[j])
    L=np.array(L)
    vect=L.transpose()
    return (K, vect)

def barycentres(vect,ListeDindice,K,n):
    res=np.zeros((K,len(vect[0])))
    coef=[0 for j in range(K)]
    for i in range(n):
        res[ListeDindice[i]] = res[ListeDindice[i]] + vect[i]
        coef[ListeDindice[i]] += 1
    for j in range (K):
        if coef[j]!=0:
            res[j]=res[j]/coef[j]
        else :
            res[j,0]=np.inf
    return res



def K_means1(K, vect,n):  # vect liste de vecteurs propres du laplacien, vect possède n lignes de taille k
    DEP = [] #Indice des sommets de départs
    for k in range(K):
        DEP.append(vect[rd.randint(0,n-1)]) #On génère aléatoirement des indices de départ #ça serait bien de force que ça prenne des points différents
    dep = np.array(DEP)
    nL=[0]
    ListeDindice = [] #Dans cette liste ListeDindice[i]=j si le sommet i appartient à la communauté j (si le vecteur représentant un communauté j est le proche de l'élément i)
    while nL !=ListeDindice: #On regarde si la liste a changée
        nL=ListeDindice
        ListeDindice=[]
        for i in range(n):
            p = 0
            mini = np.linalg.norm(dep[0]-vect[i],2)
            for j in range(1,K):                        #On cherche quel "vecteur de communauté" est le plus proche de l'élément i
                nor =np.linalg.norm(dep[j]-vect[i],2)
                if nor < mini:
                    mini = nor
                    p = j
            ListeDindice.append(p)
        dep=barycentres(vect, ListeDindice, K, n)
    s=0
    for i in range(n):
        s+=np.linalg.norm(dep[ListeDindice[i]]-vect[i],2)

    Com=[[] for j in range(K)]
    for i in range(n):
        Com[ListeDindice[i]].append(i+1)
    return Com,s

def Opti_Kmeans(K,vect,n,iteration):
    min=np.inf
    ComFinal=[]
    for i in range(iteration):
        Com,s=K_means1(K,vect,n)
        if s<min:
            ComFinal=[list(Com[j]) for j in range(K)]
            min =s
    return ComFinal


####################################################################################

"""n=30
k=4
G = GraphSBM(n, k)
G.generer_SSBM(.9, 0.1,1)
#G.afficher()
G.histogramme()

(K, vect) = spectral_clustering_avec_k(G.Adj,k)
print (K)
com=Opti_Kmeans(K,vect,n,10)
print(len(com))
print(com)
G.trac_graph_communaute(com)"""


## Non backtracking matrix

def NonBacktrac(Adj,n):
    L=[]
    for i in range(n):
        for j in range(n):
            if Adj[i,j]==1:
                L.append((i,j))

    nE=len(L)
    B=np.zeros((nE,nE))
    for i,e1 in L:
        for j,e2 in L:
            if e1[1]==e2[0] and e1[0]!=e2[1]:
                B[i,j]=1
    return B
#Bethe-Hessian
def Bethe_Hess(Adj,r):
    D = np.diag(Adj.sum(axis=0))
    return (r*r-1)*np.diag(np.ones(n))+D-r*Adj


def Bethe_Hess_weak_recovery(Adj,k,iteration):
    rc=np.trace(np.diag(Adj.sum(axis=0)))/len(Adj)
    Hplus=Bethe_Hess(Adj,rc)
    Hmoins=Bethe_Hess(Adj,-rc)
    (Vap_p,Vep_p)=Vap_Vep(Hplus)
    (Vap_m, Vep_m) = Vap_Vep(Hmoins)
    L = []
    i=0
    Vep_p = Vep_p.transpose()
    Vep_m = Vep_m.transpose()

    while(Vap_p[i]<0):
        L.append(Vep_p[i])
        i+=1
    i=0
    while(Vap_m[i]<0):
        L.append(Vep_m[i])
        i+=1
    L = np.array(L)
    vect=L.transpose()
    print(len(vect[0]))
    com=Opti_Kmeans(k,vect,len(Adj),iteration)
    return(com)
def Improved_BH_com_detect(Adj):
    D=np.diag(Adj.sum(axis=0))
    c_phi = np.sqrt(np.trace(D*D)/D.sum())
    k=0
    H=Bethe_Hess(Adj,c_phi)
    (Vap,Vep)=Vap_Vep(Adj)
    while(Vap[k]<0):
        k+=1
    print(k)
    #On fait commencer r à 1

n=100
k=2
G = GraphSBM(n, k)
G.generer_SSBM(.9, 0.1,1)
Improved_BH_com_detect(G.Adj)

#com=Bethe_Hess_weak_recovery(G.Adj,k,k*10)
#G.trac_graph_communaute(com)




