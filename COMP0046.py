#!/usr/bin/env python
# coding: utf-8

# In[215]:


import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import requests
import pandas as pd
import scipy


# In[216]:


exposure=pd.read_csv("interbankExposures.csv",header=None)


# In[217]:


ex=np.array(exposure)


# In[218]:


adj_matrix = np.where(exposure != 0, 1, 0)


# In[219]:


G1=nx.DiGraph(adj_matrix)


# In[220]:


G1.edges(data=True)


# In[221]:


## Q1


# In[224]:


def degree_histogram_directed(G, in_degree=False, out_degree=False):
    """Return a list of the frequency of each degree value.

    Parameters
    ----------
    G : Networkx graph
       A graph
    in_degree : bool
    out_degree : bool

    Returns
    -------
    hist : list
       A list of frequencies of degrees.
       The degree values are the index in the list.

    Notes
    -----
    Note: the bins are width one, hence len(list) can be large
    (Order(number_of_edges))
    """
    nodes = G.nodes()
    if in_degree:
        in_degree = dict(G.in_degree())
        degseq=[in_degree.get(k,0) for k in nodes]
    elif out_degree:
        out_degree = dict(G.out_degree())
        degseq=[out_degree.get(k,0) for k in nodes]
    else:
        degseq=[v for k, v in G.degree()]
    dmax=max(degseq)+1
    freq= [ 0 for d in range(dmax) ]
    for d in degseq:
        freq[d] += 1
    return freq


# In[227]:


# #G = nx.scale_free_graph(5000)
# in_degree_freq = degree_histogram_directed(G1, in_degree=True,out_degree=True)
# #out_degree_freq = degree_histogram_directed(G, out_degree=True)
# degrees = range(len(in_degree_freq))
# plt.figure(figsize=(12, 8)) 
# plt.plot(range(len(in_degree_freq)), in_degree_freq, 'go-', label='in-degree') 
# #plt.loglog(range(len(out_degree_freq)), out_degree_freq, 'bo-', label='out-degree')
# plt.xlabel('Degree')
# plt.ylabel('Frequency')


# In[226]:


type(nx.degree_histogram(G1))
plt.plot([i/145 for i in nx.degree_histogram(G1)])


# In[228]:


## clustering


# In[229]:


nx.clustering(G1,nodes=None,weight=None)


# In[230]:


nx.degree_pearson_correlation_coefficient(G1)
#disassortivity


# In[231]:


print(nx.average_clustering(G1))


# ## balance sheet- asset/liability/equity

# In[233]:


equity=pd.read_csv("bankEquities.csv",header=None)


# In[234]:


investment=pd.read_csv("bankAssetWeightedNetwork.csv",header=None)


# In[235]:


ss=equity[:1].values.tolist()


# In[236]:


equity_ready=list(np.ravel(ss))
#equity_ready
equity_ready


# In[237]:


total_investment=np.sum(investment, axis=1)
#print(type(total_lend))
total_lend=np.sum(exposure, axis=1)
print(total_investment)


# In[238]:


leverage=[]
for i in range(145):
    leverage.append(( total_investment[i]+total_lend[i])/equity_ready[i])


# In[239]:


liability=[]
for i in range(145):
    liability.append(total_investment[i]+total_lend[i]-equity_ready[i])
#liability


# In[240]:


max(liability)


# In[241]:


#find which comp has max liability
b=liability.index(2160523352.0)
print(b)


# ## erdos-renyi

# In[242]:


#prefer edges
eros=nx.gnm_random_graph(145, 6191, seed=1234, directed=True)


# In[243]:


plt.plot(nx.degree_histogram(eros))


# In[244]:


import matplotlib.pyplot as plt
import numpy as np

plt.plot(nx.degree_histogram(eros),label='Erdős-Rényi')
plt.plot(nx.degree_histogram(G1),label='Exposure')
plt.xlabel('degree') 
plt.ylabel('frequency') 
  
# displaying the title
plt.title("Degree distribution: Exposure VS Erdős-Rényi")
plt.legend() 
plt.show()


# In[245]:


nx.degree_pearson_correlation_coefficient(eros)
#neutral


# In[246]:


print(nx.average_clustering(eros))
#real world shows a higher level of clustering


# ## balance sheet ccdf & cdf

# In[267]:


#investment=investment.to_numpy()
equity2=equity.to_numpy()
#investment
print(type(equity2))


# In[268]:


cdf = np.sort(equity2).cumsum() / equity2.sum()
ccdf = 1 - cdf
plt.loglog(ccdf)
plt.xlabel('log_10(equity)') 
plt.ylabel('log_10(1-CDF)') 
  
# displaying the title
plt.title("CCDF graph for equities")
  
#资产规模大的公司很少
#non-zero enry of 
#discte


# In[269]:


plt.loglog(cdf)


# ## interbank exposure

# In[250]:


ex_cdf = np.sort(ex).cumsum() / ex.sum()
ex_ccdf = 1 - ex_cdf
plt.loglog(ex_ccdf)
plt.xlabel('log_10(exposure)') 
plt.ylabel('log_10(1-CDF)') 
  
# displaying the title
plt.title("CCDF graph for interbank exposure")
  


# In[251]:


from scipy.stats import kurtosis, skew

mean = np.mean(equity_ready)
variance = np.var(equity_ready)
skewness = skew(equity_ready)
kurtosis = kurtosis(equity_ready)


# In[252]:


print(mean)
print(variance)
print(skewness)
print(kurtosis)


# ## Q2

# In[253]:


def default(equity, default_matrix):
    for i in np.arange(len(equity)):
        if equity[i] < 0:
            default_matrix[i] = 1
    return default_matrix
# look at the equity matrix which default, all less than 0 are seen as default, and then write 1


# In[254]:


def update_e(r,default_matrix,ext_asset,exposure,liability):
    empty1 = np.ones(145)
    e = ext_asset + ((empty1-default_matrix)*exposure).sum(axis=1) + (r*default_matrix*exposure).sum(axis=1) - liability
    return e


# In[255]:


def furfine(r,start,ext_asset,exposure,equity,liability):
    default_matrix = np.zeros(145)#Assume that none of the default
    
    equity[start] = -1e10#Assume that one defaults
    
    i=0
    
    while i < 145:
        default_matrix = default(equity,default_matrix)
        equity = update_e(r,default_matrix,ext_asset,exposure,liability)    
        i=i+1
        
    return default_matrix


# In[256]:


ext_asset = investment.sum(axis=1)
int_asset = exposure.sum(axis=1)
equity_matrix = equity.sum(axis=0)


ext_asset1 = investment.sum(axis=1)
int_asset1 = exposure.sum(axis=1)
equity1 = equity.sum(axis=0)
liability1 = ext_asset+int_asset-equity_matrix

furfine(0,82,ext_asset1,exposure,equity1,liability1)

sum(furfine(0,82,ext_asset1,exposure,equity1,liability1))


# In[257]:


f=[]
for i in range(145):
    ext_asset = investment.sum(axis=1)
    int_asset = exposure.sum(axis=1)
    equity_matrix = equity.sum(axis=0)


    ext_asset1 = investment.sum(axis=1)
    int_asset1 = exposure.sum(axis=1)
    equity1 = equity.sum(axis=0)
    liability1 = ext_asset+int_asset-equity_matrix
    
    x=sum(furfine(0.2, i, ext_asset1, exposure, equity1, liability1))
    #x=furfine(0.2, i, ext_asset1, exposure, equity1, liability1)
    f.append(x)


# In[258]:


a=f.index(113)
print(a)


# In[259]:


x=range(145)
plt.plot(x,f)
plt.title('Furfine algorithm,r=0.2')
plt.xlabel('Firms')
plt.ylabel('number of defaults it may affect')


# ## Q3

# In[260]:


def portfolio_val(j, investment,a):
    #initial setup
    p_i = np.ones(20)
    q_i = np.zeros(20)
    d_i = np.zeros(20)#
    
    for i in range(20):
        if investment[i][j] != 0:
            d_i[i] = 1
            
    for i in range(20):   
        if d_i[i] == 1:
            q_i[i] = investment[i][j]/investment[i].sum()
    
    p_i = p_i*(1-a*q_i)
    
    return q_i


# In[261]:


def find_p(e, investment,a):
    
    D= np.zeros(shape=(145,20))
    
    for j in range(145):
        if e[j] < 0:
            for i in range(20):
                if investment[i][j] > 0:
                    D[j][i] = 1  
    Q =(D*investment).sum(axis=0)/investment.sum(axis=0)
    P =(1-a*Q)
    
    return P


# In[262]:


def furfine_overlap(r,start,exposure,equity,liability,investment,a):
    equity[start] = -1e10 #assume first one defaults
    default_matrix = default(equity, np.zeros(145))
    p_i = find_p(equity, investment,a)    
    empty1 = np.ones(145)
    
    i=0
    
    while i < 145:
        default_matrix = default(equity,default_matrix)
        equity = ((empty1 - default_matrix)* exposure).sum(axis=1) + (r * default_matrix * exposure).sum(axis=1) + (p_i * investment).sum(axis=1) - liability
        
        p_i = find_p(equity, investment,a)    
        i=i+1

        
    return int(default_matrix.sum())


# In[264]:


ol=[]
for i in range(145):
    ext_asset = investment.sum(axis=1)
    int_asset = exposure.sum(axis=1)
    equity_matrix = equity.sum(axis=0)


    ext_asset1 = investment.sum(axis=1)
    int_asset1 = exposure.sum(axis=1)
    equity1 = equity.sum(axis=0)
    liability1 = ext_asset+int_asset-equity_matrix

    #x=sum(furfine(0.2, i, ext_asset1, exposure, equity1, liability1))
    x=furfine_overlap(0.2,i,exposure,equity1,liability1,investment,0.2)
    ol.append(x)


# In[265]:


o1=ol[27:].index(139)
print(o1)


# In[266]:



x=range(145)
plt.plot(x,ol)
plt.title('Furfine algorithm overlaping,r=0.2,α=0.2')
plt.xlabel('Banks')
plt.ylabel('number of defaults it may affect')


# In[ ]:


ol2=[]
for i in range(145):
    ext_asset = investment.sum(axis=1)
    int_asset = exposure.sum(axis=1)
    equity_matrix = equity.sum(axis=0)


    ext_asset1 = investment.sum(axis=1)
    int_asset1 = exposure.sum(axis=1)
    equity1 = equity.sum(axis=0)
    liability1 = ext_asset+int_asset-equity_matrix

    #x=sum(furfine(0.2, i, ext_asset1, exposure, equity1, liability1))
    x=furfine_overlap(0.6,i,exposure,equity1,liability1,investment,0.8)
    ol2.append(x)

