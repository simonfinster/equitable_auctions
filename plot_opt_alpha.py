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

    nbBuyers, nbItems = 3, 2
    cs = np.linspace(0, 1, 101)
    alphas = np.linspace(0, 1, 101)
    gamma = 0.65

    # !! IMPORTANT !! distribution must be bounded with positive pdf

    distrib = scipy.stats.uniform(0, 1)
    distrib_name = "uniform"
    
    clb = np.linspace(0, 1, 100)
    alphalb = (2*nbBuyers*(1-clb))/(2*nbBuyers-clb*(nbBuyers-2))

    # distrib = scipy.stats.truncnorm(-2, 2, loc=2, scale=1)
    # distrib_name = "normal"
    # clb = np.linspace(0, 1, 100)
    # alphalb = (1-clb)

    # distrib = scipy.stats.truncexpon(3)
    # distrib_name = "expon"
    # clb = np.linspace(0, 1, 100)
    # alphalb = (2*nbBuyers*(1-clb))/(2*nbBuyers-clb*(nbBuyers-(nbItems+1)))

    # distrib = scipy.stats.beta(0.5, 0.5)
    # distrib_name = "beta0.5-0.5"

    instances,auctions,cmap,linestyles,mycolors,description = plot_config.config(distrib,distrib_name,nbBuyers,nbItems,cs,alphas,mpl,plt)

    # Plot signal pdf
    with PdfPages('figures/signal_pdf_%s.pdf' % description) as pdf:    
        fig = plt.figure(figsize=(5,5), tight_layout=True)
        axes = [fig.subplots(ncols=1)]
        Q = np.linspace(0,1,100)
        X = distrib.isf(1-Q)
        Y = distrib.pdf(X)
        axes[0].plot(X,Y,color=mycolors[0],linewidth=3,label=r'signal pdf')
        axes[0].set_xlabel("Signal $s$")
        axes[0].set_ylabel("$f(s)$")
        axes[0].legend()
        pdf.savefig(fig)
        plt.close()

    # WEV optimal alpha figure in paper
    evarw = np.array([[auctions[j].evar(instances[i], True)
        for j in range(len(auctions))]
        for i in range(len(instances))])
    eminiw = np.argmin(evarw, axis=1)
    monotonicity = np.array([[auctions[j].monotonicity(instances[i])
        for j in range(len(auctions))]
        for i in range(len(instances))])
   
    with PdfPages('Figures for EC/optimal_alpha_%s.pdf' % description) as pdf:    
        fig = plt.figure(figsize=(6.5,5), tight_layout=True)
        axes = fig.subplots(ncols=2,
            gridspec_kw={"width_ratios":[.95,.05]})
        im = axes[0].imshow(evarw.T, aspect="auto", origin="lower", cmap = cmap,
            norm=mpl.colors.PowerNorm(gamma=0.35),
            extent=(-.05/(len(instances)-1), 1+.05/(len(instances)-1),
                    -.05/(len(auctions)-1), 1+.05/(len(auctions)-1)))
        con = axes[0].contour(monotonicity.T, levels=[0.99], origin="lower",
            extent=(-.05/(len(instances)-1), 1+.05/(len(instances)-1),
                    -.05/(len(auctions)-1), 1+.05/(len(auctions)-1)),
            linewidths=3, colors=mycolors[0])
        p = con.collections[0].get_paths()[0]
        v = p.vertices
        x = v[:,0]
        y = v[:,1]
        x2 = [0, min(x)]
        axes[0].fill_between(x,0,y, color="none", hatch="/", edgecolor=mycolors[0], linewidth=0.0)
        axes[0].fill_between(x2,0,1, color="none", hatch="/", edgecolor=mycolors[0], linewidth=0.0)
        axes[0].plot(cs, alphas[eminiw], ":", color=mycolors[9], linewidth=3, label=r'$\alpha^*(c)$')
        axes[0].plot(clb, alphalb, linewidth=3, linestyle='--', color=mycolors[2], label=r'$\alpha$-lb')
        axes[0].set_xlabel("$c$")
        axes[0].set_ylabel("$\\alpha$")
        axes[0].set_xlim(-.3/(len(instances)-1), 1+.3/(len(instances)-1))
        axes[0].set_ylim(-.3/(len(auctions)-1), 1+.3/(len(auctions)-1))
        proxy = [plt.Line2D([],[], linewidth=3, linestyle=':', color=mycolors[9]),
                 plt.Rectangle((0,0),1,1, linewidth=0.5, fill=None, hatch='//',
                            color=con.collections[0].get_edgecolor()[0]),
                 plt.Line2D([],[], linewidth=3, linestyle='--', color=mycolors[2])]
        labels = [r'$\alpha^*(c)$', "MEU holds", r'$\alpha$-lb']
        #labels = [r'$\alpha^*(c)$', "MEU holds"]
        axes[0].legend(proxy, labels, loc='center left')
        plt.colorbar(im, cmap=cmap, cax=axes[1], label="WEV")
        axes[1].yaxis.set_ticks_position('left')
        axes[1].set_yscale('function', functions=(forward, inverse))
        pdf.savefig(fig)
        plt.close()
    
