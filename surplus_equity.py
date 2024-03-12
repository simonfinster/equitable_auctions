import sys
import scipy.stats, scipy.integrate
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def orderstat(n, k, cond=[]):
    """
    Order statistics of the uniform distribution
    
    Parameters
    ----------
        n: integer
            number of items
        k: integer
            rank
        cond: array_like
            conditioning
    Returns
    -------
        distrib: scipy.stats.rv_continuous
            distribution of the k-th highest value, given conditioning
    """
    assert(1 <= k <= n)
    c = np.array([(0.,0)] + sorted(list(cond)) + [(1.,n+1)])
    i = np.searchsorted(c[1:-1,1], k)
    (x,a),(y,b) = c[i], c[i+1]
    loc, scale = x, y-x
    return scipy.stats.beta(k-a, b-k, loc=loc, scale=scale)

def integrate(f, a, b):
    return scipy.integrate.quad(f, a, b)[0]

def absrel_error(x, y):
    if isinstance(x, np.ndarray):
      return np.array([absrel_error(u,v) for u,v in zip(x,y)])
    if x == 0: return abs(y)
    if y == 0: return abs(x)
    return min(abs(x-y), abs(x-y)/(abs(x)+abs(y)))

def memorization(function):
    """
    Memorization decorator.
    """
    memory = dict()
    def inner(*args, **kwargs):
        key = args + tuple(kwargs.items())
        if key not in memory:
            memory[key] = function(*args, **kwargs)
        return memory[key]
    return inner

def vectorization(function):
    """
    Vectorization decorator wrt to the last argument.
    """
    def inner(*args):
        if isinstance(args[-1], np.ndarray):
            return np.array([function(*args[:-1], x) for x in args[-1]])
        return function(*args)
    return inner
    

def normalization(function):
    """
    Normalization decorator wrt to the last argument.
    """
    def inner(*args, normalize=None):
        res = function(*args)
        if normalize is not None:
            res = res * normalize(args[-1])
        return res
    return inner


def interpolation(inf=0, sup=1, epsilon=1e-2,
    interpolator=scipy.interpolate.PchipInterpolator):
    def wrapper(function):
        """
        Interpolation decorator wrt to the last argument.
        """
        memory = dict()
        def inner(*args, **kwargs):
            args, v = tuple(args[:-1]), args[-1]
            key = args + tuple(kwargs.items())
            """
            def evaluate(l, r, vl, vr, res=[]):
                if res == []:
                    res.append((l,vl))
                error = absrel_error(l,r)
                if error < epsilon:
                    res.append((r,vr))
                else:
                    m, vm = (l+r)/2, function(*args, (l+r)/2, **kwargs)
                    evaluate(l, m, vl, vm, res)
                    evaluate(m, r, vm, vr, res)
                return res
            """
            if key not in memory:
                print("interpolating", function, key)
                """
                vinf = function(*args, inf, **kwargs)
                vsup = function(*args, sup, **kwargs)
                x, y = zip(*evaluate(inf, sup, vinf, vsup))
                """
                x = np.linspace(inf, sup, int(1/epsilon))
                y = function(*args, x, **kwargs)
                memory[key] = interpolator(x,y)
                ex = np.array([(x[i-1]+x[i])/2 for i in range(1, len(x))])
                ey = function(*args, ex, **kwargs)
                error = absrel_error(ey, memory[key](ex))
                print("- error max =", error.max())
                print("- error avg =", error.mean())
                """
                plt.figure()
                plt.title(str(function))
                plt.plot(x, y, ".")
                X = np.linspace(inf, sup, 5*len(x))
                plt.plot(X, memory[args](X))
                plt.show()
                #"""
            if v is None:
                return memory[key]
            return memory[key](v)
        return inner
    return wrapper

@memorization
def expect(distrib, f=None, lb=None, ub=None, cond=False):
    return distrib.expect(func=f, lb=lb, ub=ub, conditional=cond)

def square(x): return x**2

class Instance:
    """
    Class that contains parameters that do not depend on the auction format.
    Any consequence of revenue equivalence goes here.
    """
    def __init__(self, distrib, common, nb_agents, nb_items):
        """
        Parameters
        ----------
            distrib: scipy.stats.rv_continuous
                Distribution of buyers' signals
            common: float
                Common value parameter, between 0 and 1,
                0 corresponds to private, 1 corresponds to common.
            nb_agents: integer
                Number of agents (at least 2)
            nb_items: integer
                Number of items (between 1 and number of agents)
        """
        self.d = distrib
        self.c = common
        self.n = nb_agents
        self.k = nb_items
        self.g = orderstat(self.n-1, self.n-self.k)
    
    def sig_of_pr(self, x):
        """
        Parameters
        ----------
            x: array_like
                Probability of winning a duel vs another agent.
        Returns
        -------
            s: array_like
                Signals, drawn from distribution.
        """
        return self.d.isf(1-x)
    
    @vectorization
    def dsig_of_pr(self, x):
        """
        Parameters
        ----------
            x: array_like
                Probability of winning a duel vs another agent.
        Returns
        -------
            deriv: array_like
                Derivative of sig_of_pr.
        """
        return 1./(1e-10+self.d.pdf(self.d.isf(1-x)))
    
    def pr_of_sig(self, s):
        """
        Parameters
        ----------
            s: array_like
                Signals, drawn from distribution.
        Returns
        -------
            x: array_like
                Probability of winning a duel vs another agent.
        """
        return self.d.cdf(s)
    
    @interpolation()
    @normalization
    def vx_of_pr(self, x):
        """
        Parameters
        ----------
            x: array_like
                quantile of one agent
        Returns
        -------
            vx: array_like
                Value induced by a buyer's signal.
        """
        return (1-self.c+self.c/self.n) * self.sig_of_pr(x)
    
    @interpolation()
    @normalization
    def dvx_of_pr(self, x):
        """
        Parameters
        ----------
            x: array_like
                quantile of one agent
        Returns
        -------
            dvx: array_like
                Derivative of vx_of_pr(x)
        """
        return (1-self.c+self.c/self.n) * self.dsig_of_pr(x)
    
    @interpolation()
    @normalization
    @vectorization
    def vy_of_pr(self, y):
        """
        Parameters
        ----------
            y: array_like
                k-th highest quantile among other agents
        Returns
        -------
            vy: array_like
                Expected value induced by other agents' signals
        """
        assert(0 <= y <= 1)
        t = self.sig_of_pr(y)
        lo = expect(self.d, ub=t, cond=True) if y > 0 else t
        hi = expect(self.d, lb=t, cond=True) if y < 1 else t
        return (self.c/self.n) * (t + hi*(self.k-1) + lo*(self.n-self.k-1))
    
    @interpolation()
    @normalization
    @vectorization
    def dvy_of_pr(self, y):
        """
        Parameters
        ----------
            x: array_like
                quantile of one agent
        Returns
        -------
            dvy: array_like
                Derivative of vy_of_pr(x).
        """
        
        assert(0 <= y <= 1)
        
        t = self.sig_of_pr(y)
        lo = expect(self.d, ub=t, cond=True) if y > 0 else t
        hi = expect(self.d, lb=t, cond=True) if y < 1 else t
        
        dt = self.dsig_of_pr(y) # !! we assume pdf is positive !! 
        dlo = (t-lo)/y if y > 0 else dt/2
        dhi = (hi-t)/(1-y) if y < 1 else dt/2
        
        return (self.c/self.n) * (dt + dhi*(self.k-1) + dlo*(self.n-self.k-1))
    
    def v_of_pr(self, x):
        """
        Parameters
        ----------
            x: array_like
        Returns
        -------
            v: array_like
        """
        return self.vx_of_pr(x) + self.vy_of_pr(x)
    
    def dv_of_pr(self, x):
        """
        Parameters
        ----------
            x: array_like
                quantile of one agent
        Returns
        -------
            dv: array_like
                Derivative of v_of_pr(x).
        """
        return self.dvx_of_pr(x) + self.dvy_of_pr(x)
    
    @interpolation()
    @vectorization
    def util_of_pr(self, x):
        """
        Parameters
        ----------
            x: array_like
                quantile of one agent
        Returns
        -------
            val: array_like
                Expected surplus of a buyer with quantile x
        """
        if x*self.dvx_of_pr(x) > 10*self.vx_of_pr(x):
            # vx is too steep at x, so interpolation of dvx is not accurate.
            valx = self.vx_of_pr(x) * self.g.cdf(x)
            payx = self.vx_of_pr(None, normalize=self.g.pdf).antiderivative()(x)
            return valx - payx
        else:
            """
            util_of_pr(x) simplifies into valx - payx where:
            valx = G(x) vx_of_pr(x)
            integration by parts gives the formula
            payx = int_0^x vx_of_pr(y) g(y) dy
                 = G(x) vx_of_pr(x) - int_0^x dvx_of_pr(y) G(y) dy
            """
            return self.dvx_of_pr(None, normalize=self.g.cdf).antiderivative()(x)
    
    @interpolation()
    @vectorization
    def __var_of_pr(self, y):
        """
        auxiliary function for var_of_pr: integrand
        """
        assert(0 <= y <= 1)
        t = self.sig_of_pr(y)
        lo = expect(self.d, ub=t, cond=True) if y > 0 else t
        hi = expect(self.d, lb=t, cond=True) if y < 1 else t
        lo2 = expect(self.d, f=square, ub=t, cond=True) if y > 0 else t**2
        hi2 = expect(self.d, f=square, lb=t, cond=True) if y < 1 else t**2
        return self.g.pdf(y) * (self.c/self.n) * (
            (hi2 - hi**2) * (self.k-1) +
            (lo2 - lo**2) * (self.n-self.k-1))
    
    def var_of_pr(self, x):
        """
        Parameters
        ----------
            x: array_like
                quantile of one agent
        Returns
        -------
            var: array_like
                Expected (over y) variance of the value received
        """
        return self.__var_of_pr(None).antiderivative()(x)
    
    @memorization
    def mean_util(self):
        """
        Returns
        -------
            var: float
                Expected surplus
        """
        return self.util_of_pr(None).antiderivative()(1)
    
    @memorization
    def var_util(self, winner):
        """
        Returns
        -------
            var: float
                Variance of the interim surplus
        """
        if winner:
            dvG = self.dvx_of_pr(None, normalize=self.g.cdf)
            dv = self.dvx_of_pr(None)
            def f(x):
                if dvG(x) == 0.:
                    return 0
                elif x*dv(x) > 10*self.vx_of_pr(x):
                    return self.n/self.k*self.util_of_pr(x)**2/self.g.cdf(x)
                else:
                    return self.n/self.k*dvG.antiderivative()(x)**2/dvG(x)*dv(x)
            return integrate(f, 0, 1) - (self.n/self.k*self.mean_util())**2
        else:
            f = lambda x: self.util_of_pr(x)**2
            return integrate(f, 0, 1) - self.mean_util()**2
    
    @memorization
    def var_val(self, winner):
        """
        Returns
        -------
            var: float
                Variance of value.
        """
        if winner:
            return self.n/self.k*self.__var_of_pr(None).antiderivative(2)(1)
        else:
            return self.__var_of_pr(None).antiderivative(2)(1)
        
    

class Auction:
    """
    Class that that contains quantities specific to each auction format.
    """
    def __init__(self, alpha):
        """
        Parameters
        ----------
            alpha: float
                alpha parameter, between 0 and 1,
                0 corresponds to uniform, 1 corresponds to pay-as-bid.
        """
        self.alpha = alpha
    
    @interpolation()
    def __bid_of_pr(self, instance, x):
        """
        auxiliary function for bid_of_pr: integrand
        """
        return instance.dv_of_pr(x) * instance.g.cdf(x) ** (1/self.alpha)
    
    @interpolation()
    @vectorization
    def bid_of_pr(self, instance, x):
        """
        Parameters
        ----------
            instance: Instance
                Description of buyers and items
            x: array_like
                quantile of one agent
        Returns
        -------
            bid: array_like
                bid of an agent with proba x
        """
        if self.alpha == 0:
            return instance.v_of_pr(x)
        else:
            f = self.__bid_of_pr(instance, None)
            if f(x) == 0.:
                # x is too small, approximate...
                return instance.v_of_pr(x)
            #elif x*instance.dv_of_pr(x)>10*instance.v_of_pr(x):
            elif f(x)<1e-6 or x*instance.dv_of_pr(x)>10*instance.v_of_pr(x):
                #print("too steep", x)
                # v is too steep at x, so interpolation of dv is not accurate.
                a, g = self.alpha, instance.g
                f = lambda y: instance.v_of_pr(y) * g.cdf(y) ** (1/a-1)
                return expect(g, f=f, ub=x) / (a * g.cdf(x) ** (1/a))
            else:
                # fast computation
                if x == 1: print(f(x))
                F = f.antiderivative() # integrate the interpolation
                return instance.v_of_pr(x) - instance.dv_of_pr(x)*F(x)/f(x)
    
    def monotonicity(self, instance):
        """
        Returns
        -------
            sign: -1, 0 or 1.
        """
        X = np.linspace(0, 1, 100)
        Y = (1-instance.c)*instance.sig_of_pr(X) \
          - self.alpha*self.bid_of_pr(instance, X)
        if np.all(Y[:-1] <= Y[1:]): # is sorted
            return 1
        if np.all(Y[:-1] >= Y[1:]): # is reversed
            return -1
        return 0
    
    def p_of_pr(self, instance, x, y):
        """
        Parameters
        ----------
            x: array_like
                quantile of one agent
            y: array_like
                k-th highest proba among other agents
        Returns
        -------
            val: array_like
                Expected payment, conditionned on x and y.
        """
        bx = self.alpha * self.bid_of_pr(instance, x)
        by = (1-self.alpha) * self.bid_of_pr(instance, y)
        return bx + by
    
    @interpolation()
    def __ivar_of_pr_b(self, instance, y):
        """
        auxiliary function for ivar_of_pr: b
        """
        return instance.g.pdf(y) * (instance.vy_of_pr(y)
            - (1-self.alpha) * self.bid_of_pr(instance, y))
    
    @interpolation()
    def __ivar_of_pr_b2(self, instance, y):
        """
        auxiliary function for ivar_of_pr: b
        """
        return instance.g.pdf(y) * (instance.vy_of_pr(y)
            - (1-self.alpha) * self.bid_of_pr(instance, y)) ** 2
    
    @interpolation()
    def ivar_of_pr(self, instance, x):
        """
        Parameters
        ----------
            instance: Instance
                Description of buyers and items
            x: array_like
                quantile of one agent
        Returns
        -------
            var: array_like
                Interim variance of surplus of a buyer, conditioned on x
        """
        a = (instance.vx_of_pr(x) - instance.util_of_pr(x)
            - self.alpha * self.bid_of_pr(instance, x))
        var = (instance.g.cdf(x) * a**2
            + self.__ivar_of_pr_b2(instance, None).antiderivative()(x)
            + 2*a*self.__ivar_of_pr_b(instance, None).antiderivative()(x))
        return (instance.var_of_pr(x) + var +
            (1-instance.g.cdf(x)) * instance.util_of_pr(x)**2)
        """
        def f(y):
            return instance.g.pdf(y) * (
                instance.vx_of_pr(x)
                + instance.vy_of_pr(y)
                - self.alpha * self.bid_of_pr(instance, x)
                - (1-self.alpha) * self.bid_of_pr(instance, y)
                - instance.util_of_pr(x)
            ) ** 2
        var = integrate(f, 0, x)
        """
    
    @memorization
    def xvar(self, instance, winner):
        """
        Ex-ante variance of one bidder
        """
        res = (instance.var_util(False) +
            self.ivar_of_pr(instance, None).antiderivative()(1))
        """
        We computed res = xvar.
            - xvar = E[u^2] - E[u]^2
            - xvarw = E[u^2|w] - E[u|w]^2
            - E[u] = k/n*E[u|w]
            - E[u^2] = k/n*E[u^2|w]
        Solving for xvarw:
            - xvarw = n/k*xvar - n/k*(n/k-1)*E[u]^2
        """
        if winner:
            n, k = instance.n, instance.k
            res = res*n/k - n/k*(n/k-1)*instance.mean_util()**2
        return res

    @memorization
    def evar(self, instance, winner):
        """
        Expected empirical variance among winners.
        """
        def aux1(x):
            val = (1 - instance.c) * instance.sig_of_pr(1-x)
            pay = self.alpha * self.bid_of_pr(instance, 1-x)
            return val - pay
        aux2 = interpolation()(aux1)(None).antiderivative()
        
        n, k = instance.n, instance.k
        f = orderstat(n-1, n-k)
        g = orderstat(n-2, n-k)
        
        a = integrate(lambda t: f.cdf(t) * aux1(1-t)**2, 0, 1)
        b = integrate(lambda t: g.pdf(t) * aux2(1-t)**2, 0, 1)
        
        res = n/k * a - n*(n-1)/(k*(k-1)) * b
        """
        We computed res = evarw.
            - evar = E[u1^2] - E[u1*u2]
            - evarw = E[u1^2|w1] - E[u1*u2|w1&w2]
            - E[u1^2] = k/n*E[u1^2|w]
            - E[u1*u2] = (k*(k-1))/(n*(n-1))*E[u1*u2|w1&w2]
        Solving for evar:
            - evar = (k*(k-1))/(n*(n-1)) * evarw
                   + (1-(k*(k-1))/(n*(n-1))) * E[u1^2]
        where E[u1^2] = xvar + E[u1]^2.
        """
        if not winner:
            factor = (k*(k-1))/(n*(n-1))
            u2 = self.xvar(instance, False) + instance.mean_util()**2
            res = res*factor + (1-factor)*u2
        return res