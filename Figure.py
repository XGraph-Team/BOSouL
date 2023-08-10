models = ['BOSouL', 'Jordan', 'LISN', 'NET']
settings = ['SW', 'ER', 'Cora', 'CiteSeer', 'PubMed']

means = {
    'BOSouL': [7.2,5.4,8.2,7.9,7.2],
    'Jordan': [7.7,6.4,9.1,9.4,10.9],
    'LISN': [8.2,6.6,8.8,8.5,11.9],
    'NET':  [7.7,5.7,8.8,8.2,10.2]
}

stds = {
    'BOSouL': [0.789,0.966,2.298,4.483,1.874],
    'Jordan': [1.159,1.075,2.923,3.836,0.738],
    'LISN': [0.919,0.966,1.476,3.308,1.287],
    'NET':  [0.949,0.483,1.814,3.706,1.549]
}

colors = {
    'BOSouL': 'red',
    'Jordan': 'blue',
    'LISN': 'green',
    'NET':  'cyan'
}

xs = [i for i, _ in enumerate(settings)]

# Plot bars in group of 3 for each model
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.set_xlabel('Graphs', fontsize = 15, fontweight='bold')
ax.set_ylabel('Distance from the true source set', fontsize = 15, fontweight='bold')

for i, model in enumerate(models):
    ax.bar(xs, means[model], yerr=stds[model], align='center',
           alpha=0.5, ecolor='black', color=colors[model],capsize=10, width=0.2)
    xs = [x + 0.2 for x in xs]

ax.set_xticks([0.3,1.3,2.3,3.3,4.3])
ax.set_xticklabels(settings)
ax.legend(models,fontsize=13)

plt.tight_layout()
plt.savefig("SIR.pdf",dpi=300)
