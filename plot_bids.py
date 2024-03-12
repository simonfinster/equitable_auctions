import sys
import scipy.stats, scipy.integrate
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plot_config


if __name__ == "__main__":

    nbBuyers, nbItems = 10, 4
    cs = np.array([0, 0.5, 0.8, 1])
    alphas = np.array([0, 0.5, 1])
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
    
    # Bid function figures in paper
    for i in range(len(instances)):
        with PdfPages('figures/bid_%s_c_%.2f.pdf' % (description, cs[i])) as pdf:
            mpl.style.use("seaborn-colorblind")
            X = np.linspace(0, 1, 100)
            X2 = np.linspace(0, distrib.isf(0), 100)
            # create figure
            fig = plt.figure(figsize=(5,5), tight_layout=True)
            axes = [fig.subplots()]
            for j in range(len(alphas)):
                Y = auctions[j].bid_of_pr(instances[i], X)
                axes[0].plot(distrib.isf(1-X), Y, ls=linestyles[j], color=colors[i], linewidth=3,
                             label=(r'$\alpha$ = %f' % alphas[j]).rstrip('0').rstrip('.'))
            axes[0].plot(X2, X2, ls=':', color='black')
            axes[0].set_xlabel("Signal $s$")
            axes[0].set_ylabel("Bid $\\beta^\\alpha$")
            #axes[0].set_yticklabels([])
            axes[0].set_xlim(left=0, right=distrib.isf(0))
            axes[0].set_ylim(bottom=0, top=distrib.isf(0))
            axes[0].legend()
            pdf.savefig(fig)
            plt.close()
    
