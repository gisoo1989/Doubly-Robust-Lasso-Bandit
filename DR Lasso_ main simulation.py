#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[ ]:


class LassoBandit:
    def __init__(self,q,h,lam1,lam2,d,N):
        self.Tx=np.empty((N, 0)).tolist()
        self.Sx=np.empty((N, 0)).tolist()
        self.Tr=np.empty((N, 0)).tolist()
        self.Sr=np.empty((N, 0)).tolist()
        self.q=q
        self.h=h
        self.lam1=lam1
        self.lam2=lam2
        self.d=d
        self.N=N
        self.beta_t=np.zeros((N,N*d))
        self.beta_a=np.zeros((N,N*d))
        self.n=0
        self.lasso_t=linear_model.Lasso(alpha=self.lam1) #for force-sample estimator
    
    def choose_a(self,t,x): #x is N*d-dim vector 
        if t==((2**self.n-1)*self.N*self.q+1):
            self.set=np.arange(t,t+self.q*self.N)
            self.n+=1
        if t in self.set:
            ind=list(self.set).index(t)
            self.action=ind//self.q
            self.Tx[self.action].append(x)
        else:
            est=np.dot(self.beta_t,x) #N by 1
            max_est=np.amax(est)
            self.K=np.argwhere(est>max_est-self.h/2.) # action indexes
            est2=[np.dot(x,self.beta_a[k[0]]) for k in self.K]
            self.action=self.K[np.argmax(est2)][0]
        self.Sx[self.action].append(x)
        return(self.action)            
             
    def update_beta(self,rwd,t):
        if t in self.set:
            self.Tr[self.action].append(rwd)
            self.lasso_t.fit(self.Tx[self.action],self.Tr[self.action])
            self.beta_t[self.action]=self.lasso_t.coef_
        self.Sr[self.action].append(rwd)
        lam2_t=self.lam2*np.sqrt((np.log(t)+np.log(self.N*self.d))/t)
        lasso_a=linear_model.Lasso(alpha=lam2_t)
        if t>5:
            lasso_a.fit(self.Sx[self.action],self.Sr[self.action])
            self.beta_a[self.action]=lasso_a.coef_
        
        


# In[ ]:


class DRLassoBandit2:
    def __init__(self,lam1,lam2,d,N,tc,tr,zt):
        self.x=[]
        self.r=[]
        self.lam1=lam1
        self.lam2=lam2
        self.d=d
        self.N=N
        self.beta=np.zeros(d)
        self.tc=tc
        self.tr=tr
        self.zt=zt
        
    def choose_a(self,t,x):  # x is N*d matrix
        if t<self.zt:
            self.action=np.random.choice(range(self.N))
            self.pi=1./self.N
        else:
            uniformp=self.lam1*np.sqrt((np.log(t)+np.log(self.d))/t)
            #print(uniformp)
            uniformp=np.minimum(1.0,np.maximum(0.,uniformp))
            choice=np.random.choice([0,1],p=[1.-uniformp,uniformp])
            est=np.dot(x,self.beta)
            if choice==1:
                self.action=np.random.choice(range(self.N))
                if self.action==np.argmax(est):
                    self.pi=uniformp/self.N+(1.-uniformp)
                else:
                    self.pi=uniformp/self.N            
            else:
                self.action=np.argmax(est)
                self.pi=uniformp/self.N+(1.-uniformp)
        #print(self.pi)
        self.x.append(np.mean(x,axis=0))
        #print(np.mean(Xmat,axis=0).shape)
        #print(self.x[-1])
        self.rhat=np.dot(x,self.beta)
        #print(self.rhat)
        return(self.action)            
             
     
    def update_beta(self,rwd,t):
        print(rwd)
        pseudo_r=np.mean(self.rhat)+(rwd-self.rhat[self.action])/self.pi/self.N
        if self.tr==True:
            pseudo_r=np.minimum(3.,np.maximum(-3.,pseudo_r))
        self.r.append(pseudo_r)
        print(pseudo_r)
        if t>5:
            if t>self.tc:
                lam2_t=self.lam2*np.sqrt((np.log(t)+np.log(self.d))/t) 
            lasso=linear_model.Lasso(alpha=lam2_t)
            #print(len(self.r))
            lasso.fit(self.x,self.r)
            self.beta=lasso.coef_


# In[ ]:


#simulation settings

N=100
d=100
s0=5
R=0.05
T=1000

sigma_sq=1.
rho_sq=0.7
V=(sigma_sq-rho_sq)*np.eye(N)+rho_sq*np.ones((N,N))


np.random.seed(1)


beta=np.zeros(d)
inds=np.random.choice(range(d),s0,replace=False)
beta[inds]=np.random.uniform(0.,1.,s0)


# In[ ]:


simul_n=10

cumulated_regret_Lasso=[]
cumulated_regret_DR=[]


# In[ ]:


for simul in range(simul_n):

    M1=LassoBandit(q=1,h=5,lam1=0.05,lam2=0.05,d=d,N=N)
    M3=DRLassoBandit2(lam1=1.,lam2=0.5,d=d,N=N,tc=1,tr=True,zt=10)
    RWD1=list()
    RWD3=list()
    optRWD=list()

    for t in range(T):
        x=np.random.multivariate_normal(np.zeros(N),V,d).T
        #x=np.hstack((np.ones(N).reshape(N,1),x))
        x_stack=x.reshape(N*d)
    
        err=R*np.random.randn()
        
        a1=M1.choose_a(t+1,x_stack)
        rwd1=np.dot(x[a1],beta)+err
        RWD1.append(np.dot(x[a1],beta))
        M1.update_beta(rwd1,t+1)
        

        a3=M3.choose_a(t+1,x)
        rwd3=np.dot(x[a3],beta)+err
        RWD3.append(np.dot(x[a3],beta))
        M3.update_beta(rwd3,t+1)
    
        optRWD.append(np.amax(np.dot(x,beta)))
    
        #print(t)
    cumulated_regret_Lasso.append(np.cumsum(optRWD)-np.cumsum(RWD1))
    cumulated_regret_DR.append(np.cumsum(optRWD)-np.cumsum(RWD3))
    
    


# In[ ]:


steps=np.arange(1,T+1)
plt.plot(steps,np.median(cumulated_regret_Lasso,axis=0),'r',label='Lasso Bandit')
plt.plot(steps,np.median(cumulated_regret_DR,axis=0),'b',label='DR Lasso Bandit')


plt.xlabel('Decision Point')
plt.ylabel('Cumulative Regret')
plt.title('Corr=0.7, d=100, N=100')
plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.2),fancybox=True,ncol=5)
plt.show()


# In[ ]:


np.save('C://Users/user/Desktop/Dropbox/NIPS 2019/arrays2/LassoD100N100Corr7.npy',cumulated_regret_Lasso)
np.save('C://Users/user/Desktop/Dropbox/NIPS 2019/arrays2/DRD100N100Corr7.npy',cumulated_regret_DR)

