import itertools

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
    random_pi = [rd.random() for i in range(K)]  # On gÃ©nÃ¨re alÃ©atoirement les L[i]
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
    Com = [[] for j in range(K)]
    for i in range(n):
        Com[X[i]].append(i + 1)
    return A,Com

def ErreurPetitK(n,com1,com2):
    k=len(com1)
    assert(k<=5)
    err=n

    H=list(itertools.permutations(com2))
    ListInd1=np.zeros(n+1)
    ListInd2=np.zeros(n+1)
    Ldiff=np.zeros(n+1)
    for j in range(k):
        for i in com1[j]:
            ListInd1[i]=j
    for com in H:
        s=0
        for j in range(k):
            for i in com[j]:
                ListInd2[i] = j
        Ldiff=ListInd1-ListInd2
        for i in range(n):
            if Ldiff[i]!=0.0:
                s+= 1
        if s<err:
            err=s
    return err/n

def CompteCommun(L1,L2): #Compte le nombre  d'Ã©lÃ©ments de deux listes triÃ©es
    imax=len(L1)
    jmax=len(L2)
    i=0
    j=0
    res=0
    while(i<imax and j<jmax):
        if(L1[i]==L2[j]):
            res+=1
            i+=1
            j+=1

        elif L1[i]<L2[j]:
            i+=1
        else:
            j+=1
    return res


def AgreementApproc(n,com1,com2):
    k=len(com1)
    agr=0
    L=[]
    for p in range(k):
        compte=0
        i=0
        for j in range(k):
            if not(j in L):
                r=CompteCommun(com1[p],com2[j])
                if r>compte:
                    compte=r
                    i=j
        L.append(i)
        agr+=compte
    return agr/n




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
        self.COM=[[] for i in range(self.K)]
    def afficher(self):
        print(self.Adj)

    def generer_aleatoirement_SBM(self):
        self.Pi = proba_quelconque(self.K)

        self.W = W_SBM(
            self.K)  # On gÃ©nÃ¨re W alÃ©atoirement (matrice des coeffs de bernoulli pour la matrice d'adjacence)

        self.Adj,self.COM = gen(self.n, self.K, self.W,
                       self.Pi)  # On gÃ©nÃ¨re la matrice d'adjacence Ã  partir des a et b alÃ©atoires et de random_pi

    def generer_aleatoirement_SSBM(self, max_a, max_b, alpha_n):
        self.Pi = loi_uniforme(self.K)

        self.a = rd.random() * max_a  # On gÃ©nÃ¨re a et b alÃ©atoirement
        self.b = rd.random() * max_b

        self.W = W_SSBM(self.K, self.a, self.b,alpha_n)  # On gÃ©nÃ¨re W Ã  partir des donnÃ©es alÃ©atoires

        self.Adj,self.COM = gen(self.n, self.K, self.W,
                       self.Pi)  # On gÃ©nÃ¨re la matrice d'adjacence Ã  partir des a et b alÃ©atoires et de random_pi

    def generer_SSBM(self, val_a, val_b,alpha_n):
        self.Pi = loi_uniforme(self.K)
        self.a = val_a
        self.b = val_b

        self.W = W_SSBM(self.K, self.a, self.b,alpha_n)

        self.Adj,self.COM = gen(self.n, self.K, self.W, self.Pi)

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
        'node_size'  : 300,
        'edge_color' : 'tab:grey',
        'with_labels': False
        }
        nx.draw(G,**options)
        plt.plot()
        plt.savefig("test.png")   
        plt.show()

    def histogramme(self,i):#1 Laplacian,2 Bethe_Hess,3 NonBacktrac
        if i == 2:
            rc=np.trace(np.diag(self.Adj.sum(axis=0))) / len(self.Adj)
            (Vap, Vep) = Vap_Vep(Bethe_Hess(self.Adj,rc))
            idx = np.flip(Vap.argsort()[::-1])
            Vap = Vap[idx]
            Vep = Vep[:, idx]  # Vep et Vap sont ici triÃ©es
            X=list(range(self.n//4))
            Y=[Vap[x] for x in X]
            plt.scatter(X,Y,s=20,color='r')
            plt.title("bethe hessian")
            plt.show()
        elif i==1:
            (Vap, Vep) = vp_laplacian(self.Adj)
            idx = np.flip(Vap.argsort()[::-1])
            Vap = Vap[idx]
            Vep = Vep[:, idx]  # Vep et Vap sont ici triÃ©es
            X = list(range(self.n // 4))
            Y = [Vap[x] for x in X]
            plt.scatter(X, Y, s=20, color='r')
            plt.title("laplacien ")
            plt.show()

        elif i == 3:
            (Vap, Vep) = Vap_Vep(NonBacktrac(self.Adj,self.n))
            idx = np.flip(Vap.argsort()[::-1])
            Vap = Vap[idx]
            Vep = Vep[:, idx]  # Vep et Vap sont ici triÃ©es
            X=list(range(self.n//4))
            Y=[Vap[x] for x in X]
            plt.scatter(X,Y,s=20,color='r')
            plt.title("non-backtracking")
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


def barycentre(vect):  # vect array de p vecteur Ã  n coordonnÃ©es
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


def vp_laplacian(adj):
    (x, y) = np.linalg.eig(laplace(adj))
    x=x.real                                #On prend la partie rÃ©elle pour ignorer la partie imaginaire apparaÃ®t Ã  cause des approximations de l'ordinateur
    y=y.real
    idx = np.flip(x.argsort()[::-1])
    x = x[idx]
    y = y[:, idx]  # Vep et Vap sont ici triÃ©
    return (x, y)

def spectral_clustering_sans_k(adj):
    (Vap, Vep) = vp_laplacian(adj)
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
    (Vap, Vep) = vp_laplacian(adj)
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



def K_means1(K, vect,n):  # vect liste de vecteurs propres du laplacien, vect possÃ¨de n lignes de taille k
    DEP = [] #Indice des sommets de dÃ©parts
    for k in range(K):
        DEP.append(vect[rd.randint(0,n-1)]) #On gÃ©nÃ¨re alÃ©atoirement des indices de dÃ©part #Ã§a serait bien de force que Ã§a prenne des points diffÃ©rents
    dep = np.array(DEP)
    nL=[0]
    ListeDindice = [] #Dans cette liste ListeDindice[i]=j si le sommet i appartient Ã  la communautÃ© j (si le vecteur reprÃ©sentant un communautÃ© j est le proche de l'Ã©lÃ©ment i)
    while nL !=ListeDindice: #On regarde si la liste a changÃ©e
        nL=ListeDindice
        ListeDindice=[]
        for i in range(n):
            p = 0
            mini = np.linalg.norm(dep[0]-vect[i],2)
            for j in range(1,K):                        #On cherche quel "vecteur de communautÃ©" est le plus proche de l'Ã©lÃ©ment i
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

"""n=300
k=15
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
def Vap_Vep(M):
    (x, y) = np.linalg.eigh(M)
    x=x.real                                #On prend la partie réelle pour ignorer la partie imaginaire apparaît à cause des approximations de l'ordinateur
    y=y.real
    idx = np.flip(x.argsort()[::-1])
    x = x[idx]
    y = y[:, idx]  # Vep et Vap sont ici triÃ©
    return (x, y)
    

def NonBacktrac(Adj,n):
    L=[]
    for i in range(n):
        for j in range(n):
            if Adj[i,j]==1:
                L.append([i,j])
    nE=len(L)
    B=np.zeros((nE,nE))
    for i in range(len(L)):
        for j in range(len(L)):
            if L[i][1]==L[j][0] and L[i][0]!=L[j][1]:
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
    #On fait commencer r Ã  1

n=80
k=5
G = GraphSBM(n, k)
G.generer_SSBM(.9, 0.1,1/n)

#Improved_BH_com_detect(G.Adj)
(K, vect) = spectral_clustering_avec_k(G.Adj,k)
com=Opti_Kmeans(k,vect,n,10)
G.histogramme(1)
G.histogramme(2)
NonBacktrac(G.Adj,n)
G.histogramme(3)
print(com)
print(G.COM)
print(1-AgreementApproc(n,com,G.COM))
#print(ErreurPetitK(n,com,G.COM))
G.trac_graph_communaute(com)


# Tableau pour les 3 axes
N=10
x =np.arange(1,2*N,2) # CrÃ©ation du tableau de l'axe k
print(x)
y=np.arange(100,(N+1)*100,100) # CrÃ©ation du tableau de l'axe n

print(y)
X, Y = np.meshgrid(x, y)
print(X.shape,Y.shape)

def function_z(n,k):
    G = GraphSBM(n, k)
    G.generer_SSBM(.9, 0.2,1)
    (K, vect) = spectral_clustering_avec_k(G.Adj,k)
    com=Opti_Kmeans(k,vect,n,10)
    return(1-AgreementApproc(n,com,G.COM))
Z=np.zeros((N,N))  # CrÃ©ation du tableau de l'axe z entre 
"""for i in range(N):
    for j in range(N):
        print(X[i][j],Y[i][j])
        Z[i][j]=function_z(Y[i][j],X[i][j])
print(Z.shape)


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(40, 100);
ax.plot_wireframe(X, Y, Z, color='black');
ax.set_xlabel('k')
ax.set_ylabel('n')
ax.set_zlabel('err')
plt.show()"""