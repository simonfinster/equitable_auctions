import sys
import scipy.stats, scipy.integrate
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plot_config

def forward(x):
    return x**gamma

def inverse(x):
    return x**(1/gamma)

if __name__ == "__main__":

    nbBuyers, nbItems = 10, 4
    cs = np.array([0, 0.5, 0.8, 1])
    alphas = np.linspace(0, 1, 51)
    gamma = 0.65

    # !! IMPORTANT !! distribution must be bounded with positive pdf

    distrib = scipy.stats.uniform(0, 1)
    distrib_name = "uniform"

    # distrib = scipy.stats.truncnorm(-2, 2, loc=2, scale=1)
    # distrib_name = "normal"

    # distrib = scipy.stats.truncexpon(3)
    # distrib_name = "expon"

    # distrib = scipy.stats.beta(0.5, 0.5)
    # distrib_name = "beta0.5-0.5"

    instances,auctions,cmap,linestyles,colors,description = plot_config.config(distrib,distrib_name,nbBuyers,nbItems,cs,alphas,mpl,plt)

    # WEV figures in paper
    evarw = np.array([[auctions[j].evar(instances[i], True)
        for j in range(len(auctions))]
        for i in range(len(instances))])
    eminiw = np.argmin(evarw, axis=1)
    
    with PdfPages('figures/WEV_%s.pdf' % (description)) as pdf:
        fig = plt.figure(figsize=(6,5), tight_layout=True)
        axes = [0,fig.subplots()]
        for i in range(len(instances)):
            Y = [evarw[i][j] for j in range(len(alphas))]
            axes[1].plot(alphas, Y, ls=linestyles[0], color=colors[i], linewidth=3,
                    label=(r'$c$ = %f' % cs[i]).rstrip('0').rstrip('.'))
        axes[1].set_xlabel("$\\alpha$")
        axes[1].set_ylabel("WEV")
        axes[1].set_xlim(left=0, right=1)
        axes[1].set_ylim(bottom=0)
        axes[1].set_yscale('function', functions=(forward, inverse))
        axes[1].legend(loc='upper right')
        pdf.savefig(fig)
        plt.close()
        
    
    
