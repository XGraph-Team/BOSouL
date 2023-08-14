import numpy as np
import matplotlib.pyplot as plt

# SIR

models = ['BOSouL', 'Jordan', 'LISN', 'NET']
settings = ['SW', 'ER', 'Cora', 'CiteSeer', 'PubMed']

means = {
    'BOSouL': [6.4,5.6,6.4,5.4,7.2],
    'Jordan': [7.4,6.6,6.6,6.6,10.6],
    'LISN': [7.8,6.2,7.4,8,10.8],
    'NET':  [7.8,6.4,6.8,7.8,10.2]
}

stds = {
    'BOSouL': [0.894,0.894,1.673,2.565,2.588],
    'Jordan': [0.548,1.140,1.517,1.817,1.517],
    'LISN': [0.447,1.095,2.510,3.082,1.924],
    'NET':  [0.447,0.894,1.483,1.924,1.095]
}

colors = {
    'BOSouL': 'red',
    'Jordan': 'blue',
    'LISN': 'green',
    'NET':  'olive'
}

x=np.arange(len(settings))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 7))
for i, model in enumerate(models):
    ax.bar( x + i * width, means[model], width, label=model, yerr=stds[model], capsize=10, alpha=0.8, color=colors[model], align='center', error_kw=dict(lw=2, capthick=2))

ax.set_ylim(3, 14)
ax.set_xticks(x+(len(models)-i)*width/2)
ax.set_xticklabels(settings, fontsize='xx-large')
ax.tick_params(axis='y', labelsize='xx-large')
ax.legend(fontsize='xx-large')
ax.set_xlabel('Graphs', fontsize = 15, fontweight='bold')
ax.set_ylabel('Distance from the true source set', fontsize = 15, fontweight='bold')

plt.tight_layout()
plt.savefig("SIR.pdf",dpi=300)

# SI

models = ['BOSouL', 'Jordan', 'LISN', 'NET']
settings = ['SW', 'ER', 'Cora', 'CiteSeer', 'PubMed']

means = {
    'BOSouL': [6.4,5.4,5.2,4.8,5.6],
    'Jordan': [8.4,6.0,5.2,4.6,6.6],
    'LISN': [8.2,5.4,5.2,4.8,6.6],
    'NET':  [7.2,6.0,5,4.4,7.0]
}

stds = {
    'BOSouL': [0.894,1.140,2.114,1.658,2.191],
    'Jordan': [0.894,0,2.588,2.074,1.517],
    'LISN': [1.095,0.894,2.588,2.280,1.342],
    'NET':  [0.837,1.224,2,2.074,1.732]
}

colors = {
    'BOSouL': 'red',
    'Jordan': 'blue',
    'LISN': 'green',
    'NET':  'olive'
}

x=np.arange(len(settings))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 7))
for i, model in enumerate(models):
    ax.bar( x + i * width, means[model], width, label=model, yerr=stds[model], capsize=10, alpha=0.8, color=colors[model], align='center', error_kw=dict(lw=2, capthick=2))

ax.set_ylim(2, 11)
ax.set_xticks(x+(len(models)-i)*width/2)
ax.set_xticklabels(settings, fontsize='xx-large')
ax.tick_params(axis='y', labelsize='xx-large')
ax.legend(fontsize='xx-large')
ax.set_xlabel('Graphs', fontsize = 15, fontweight='bold')
ax.set_ylabel('Distance from the true source set', fontsize = 15, fontweight='bold')

plt.tight_layout()
plt.savefig("SI.pdf",dpi=300)