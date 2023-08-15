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



# Time

models = ['BOSouL', 'Jordan', 'LISN', 'NET']
settings = ['1000', '2000', '3000', '4000', '5000']

means = {
    'BOSouL': [396.69, 772.15, 1179.77, 1599.77, 2018.98],
    'Jordan': [28.10, 134.85, 261.57, 410.85, 580.86],
    'LISN': [64.19, 332.51, 740.55, 1290.92, 1989.67],
    'NET': [40.42, 211.22, 386.23, 541.71, 704.72]
}

stds = {
    'BOSouL': [5.84, 0.96, 2.88, 12.53, 13.56],
    'Jordan': [6.05, 53.41, 63.71, 25.72, 52.32],
    'LISN': [9.20, 82.19, 93.23, 42.43, 74.27],
    'NET': [11.82, 121.84, 136.61, 81.14, 118.44]
}

colors = {
    'BOSouL': 'red',
    'Jordan': 'blue',
    'LISN': 'green',
    'NET': 'olive'
}

# Create plot
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.set_xlabel('Graph size', fontsize=15, fontweight='bold')
ax.set_ylabel('Runtime (in seconds)', fontsize=15, fontweight='bold')

for i, model in enumerate(models):
    plt.errorbar(
        settings,
        means[model],
        yerr=stds[model],
        elinewidth=3,
        capsize=6,  # Size of the caps at the end of error bars
        marker='o',  # Add markers to data points
        markersize=8,  # Size of the markers
        markeredgecolor='black',  # Marker edge color
        linewidth=1,
        color=colors[model],
        label=model
    )

ax.legend(
    loc='upper left',
    markerscale=0.6,
    facecolor=(1, 1, 1, 0.05),
    fontsize=12,
    prop=dict(weight='bold', size=10)
)

ax.tick_params(axis='both', labelsize=12)  # Adjust tick label font size

plt.title('Runtime Comparison', fontsize=18, fontweight='bold')  # Add title
plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines

plt.tight_layout()  # Improve spacing between elements
plt.savefig("time.pdf",dpi=300)
