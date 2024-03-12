import seaborn as sns
import surplus_equity as se

def config(distrib,distrib_name,nbBuyers,nbItems,cs,alphas,mpl,plt):
    instances = [se.Instance(distrib, c, nbBuyers, nbItems) for c in cs]
    auctions = [se.Auction(alpha) for alpha in alphas]
    cmap = plt.cm.get_cmap('rocket')
    linestyles = ('-','--','-.',':',(5, (10, 3)))
    colors = sns.color_palette("colorblind").as_hex()
    description = "%s_n%d_k%d" % (distrib_name, nbBuyers, nbItems)
    SMALL = 16
    MEDIUM = 18
    BIG = 20
    plt.rc('font', size=SMALL)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL)    # legend fontsize
    plt.rc('figure', titlesize=BIG)     # fontsize of the figure title
    mpl.rcParams['hatch.linewidth'] = 2.0
    return instances,auctions,cmap,linestyles,colors,description
