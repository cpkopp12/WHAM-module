"""
author: Cameron Kopp

created: 5/15/2024
"""
#IMPORTS
import numpy as np
import scipy.integrate as integrate
from numpy import random as r
import math
import matplotlib.pyplot as plt
from matplotlib import colors



# =============================================================================
# WhamConvergence_1Ion class
# =============================================================================

class WhamConvergence_1Ion:
    """
    class contains all of the methods for solving the WHAM Equations,
    going to begin by using a BFGS convergence algoritm with an Armijo
    line search
    """
    def __init__(self, xmin, xmax, biasingStepNumber, sampleNumber, binSize, springConstant,
                 fname):
        """
        

        Parameters
        ----------
        xmin : float
            min value on range xmin-xmax
        xmax : float
            max value on ranage xmin-xmax.
        biasingStepNumber : Int
            determines the number of sets of samples taken from different 
            biasing centers, evenly spaced over xmin-xmax.
        binSize : float
            size of the bins which discriteze the x coord.
        springConstant : float
            determines the strength of the biasing potential.
        
        fname : string
            name of .txt file to read data from

        Returns
        -------
        going to create arrays for bin indices and coordinates, and well as
        biasing indicies and coordinates
        going to read in and instantiate histogram from the datafile
        
        also establishes the neccessary params which will not be altered
        throughout convergence, i.e. biasing potential, summations from 
        histogram
        

        """
        
        self.xmn = xmin
        self.xmx = xmax
        self.bStepNum = biasingStepNumber
        self.N = sampleNumber
        self.k = springConstant
        self.binSize = binSize
        self.fileName = fname
        
        
        #set up two arrays, bin coordinates (center) and bin indicies
        binNumber = (self.xmx-self.xmn)/self.binSize
        #array with bin indexes = binI
        self.binI = np.arange(0,binNumber, dtype=int)
        #array with bin x coordinate values = binX
        self.binX = np.array((self.binI * self.binSize),
                               dtype=float)
        
        #set up two arrays, coordinates of the center of biasing potential
        #for each step biasX, and the indicies for the biasing steps baisI
        stepSize = (self.xmx-self.xmn)/(self.bStepNum)
        self.simI = np.arange(0, self.bStepNum, dtype=int)
        self.biasX = np.array((self.simI * stepSize),
                                dtype=float)
        
        #histogram array
        self.hist = np.zeros(np.size(self.binI))
        #number of samples per sim array
        self.N_i = np.zeros(np.size(self.simI))
        
        #need total number of samples per baising step
        #self.N_i = int(math.floor(self.N/self.bStepNum))
        
        #going to call the baising potential matix c_i_l (dim[SxM])
        self.c_i_l = self.biasingPotentialMatrix()
        
        
        #read histogram data from file
        with open(self.fileName) as myFile:
            st= myFile.read();
            
        lst = st.split()
        #isolate seperate into histgram and ni array
        histlst = lst[:np.size(self.hist)]
        nlst = lst[np.size(self.hist):]
        for i in self.binI:
            self.hist[i] = float(histlst[i])
        # for j in self.simI:
        #     self.N_i[j] = nlst[j]
        
        self.N_i += self.N/self.bStepNum
           
            
    def printHistogramFromFile(self):
        # make data
        x = self.binI
        y = self.hist
         
        # plot
        fig, ax = plt.subplots()
         
        ax.plot(x, y, linewidth=2.0)
         
        plt.show()
        
        return
    
    
    
    def biasingPotentialMatrix(self):
        """
        calculate the simIng potential matrix, dim(size(simI),size(binI)),
        the element i,l represents the value of the biasing potential from
        simulation step i at the x value in the center of bin l

        Returns
        -------
        biasing potential matrix, dim(size(biasI),size(binI)), dtype=float

        """
        
        c_i_l = np.zeros((np.size(self.simI),np.size(self.binI)),
                                       dtype = float)
        
        #loop over bias index, get x coord for each
        for i in self.simI:
            iIn = int(i)
            biasx = self.biasX[iIn]
            #loop over bin index, get x coord for each
            for l in self.binI:
                lIn = int(l)
                binx = self.binX[lIn]
                #calc elment of bias matrix
                c_i_l[iIn, lIn]  = np.exp((-1*(self.k/2))*((binx- biasx)**2))
                 
                
        return c_i_l
    
    def initialEstimates(self):
            """
            initial estimations for rho, normalization constant f,
            and the change of variable g = log(1/f),
            going to use the most simple approximation possible,
            each vector will have identical elemnts independent of
            histogram data

            Returns
            -------
            (f0, g0)

            """
            #start by assuming every bin index is equally likely
        
            f0 = np.ones((np.size(self.simI)),dtype=float)
            g0 = np.zeros((np.size(self.simI)),dtype=float)
            
            
            
            
            
            return (f0, g0)
        
        
        
    def calcA_g(self, g):
        """
        The wham equation which we aim to minimize,
        A(g1,..,gs): R^s -> R, reference pdf for full expression

        Returns
        -------
        A(g1,...,gs), scaler value

        """
        
        #initialize first two summations
        sum1 = 0
        sum2 = 0
        
        #first term
        for i in self.simI:
            sum1 = sum1 + (self.N_i[i] * g[i])
        
        #second term
        for l in self.binI:
            if(self.hist[l] != 0):
                #initialize third sum
                sum3 = 0
                for j in self.simI:
                    sum3 = sum3 + (self.N_i[j] * self.c_i_l[j,l] * np.exp(g[j]))
                    
                if(sum3 != 0):
                    sum2 = sum2 + (self.hist[l] * np.log(self.hist[l]/sum3))
        
        A_g = - sum1 - sum2
        
        return A_g
    
    
    def gradient_dA_dgi(self, g):
        """
        

        Parameters
        ----------
        g : float
            indicies i (1,S)

        Returns
        -------
        dA/dg_i : fload
            indicies i (1/S)

        """
        #initialize gradient vector
        da_dgi = np.zeros(np.size(self.simI))
        
        for i in self.simI:
            sum1 = 0
            
            for l in self.binI:
                
                if(self.hist[l] != 0):
                    sum2 = 0
                    
                    for j in self.simI:
                        sum2 = sum2 + (self.N_i[j] * self.c_i_l[j,l] *
                                       np.exp(g[j]))
                    
                    sum1 = sum1 + ((self.hist[l] * self.c_i_l[i,l])/sum2)
            
            da_dgi[i] = self.N_i[i] * ((np.exp(g[i]) * sum1) - 1)
            
        return da_dgi
    
    def armijo_LineSearch(self,gk,dA_gk,hk,alpha,tao,beta,lsil):
        """
        

        Parameters
        ----------
        gk: float array
            current value for gk
        dA_gk: float array
            current value of dA(gk)/dg_i
        Hk: float square matrix
            current value of hessian
        alpha : float
            alpha parameter in armijo line search, see pdf
        tao : float, tao < 1
            tao parameter in armijo line search, see pdf
        beta : float, beta <=1
            beta parameter in armijo line search, see pdf
        i : int
            iteration number within BFGS optimization
        lsil : int
            line search interation limit, primarily for testing purposes (will
                most likely be set as optional) 

        Returns
        -------
        gk1: float array
            the next iteration of g values resulting from the line search,
            g_(k+1)

        """
        
        #pk = descent direction
        pk = -1*np.dot(hk,dA_gk)
        tolconst = beta * np.dot(dA_gk.T,pk)
        #initial comparison
        A_gk = self.calcA_g(gk)
        Atol_l = A_gk + (alpha * tolconst)
        gk_test = gk + (alpha * pk)
        Atest_l = self.calcA_g(gk_test)
        #lsearch iteration index
        l = 0
        
        #armijo condition
        while (Atest_l > Atol_l) and (l < lsil):
            l += 1
            alpha *= tao
            gk_test = gk + (alpha * pk)
            Atest_l = self.calcA_g(gk_test)
            Atol_l = A_gk + (alpha * tolconst)
        
        gk1 = gk_test
        
        print('iterations, a_gk, a_gk1: ', l, A_gk, self.calcA_g(gk1))
        
        return gk1
    
    
    #FUNCTIONS FOR WOLFE-POWELL ------------------------
    def phi(self,g0,dgk,alpha):
        """
        

        Parameters
        ----------
        g0 : TYPE
            DESCRIPTION.
        dgk : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        return self.calcA_g(g0+(alpha*dgk))
    
    def dphi(self,g0,dgk,alpha):
        """
        

        Parameters
        ----------
        g0 : TYPE
            DESCRIPTION.
        dgk : TYPE
            DESCRIPTION.
        alpha : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        gl = g0 + (alpha*dgk)
        da_gl = self.gradient_dA_dgi(gl)
        
        return np.inner(gl,da_gl)
    
    def armijo_WP_lsearch(self,gk,dA_gk,hk,alpha,tao,beta,lsil,ep,np):
        """
        

        Parameters
        ----------
        gk : TYPE
            DESCRIPTION.
        dA_gk : TYPE
            DESCRIPTION.
        hk : TYPE
            DESCRIPTION.
        alpha : TYPE
            DESCRIPTION.
        tao : TYPE
            DESCRIPTION.
        beta : TYPE
            DESCRIPTION.
        lsil : TYPE
            DESCRIPTION.
        ep : TYPE
            DESCRIPTION.
        np : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        dgk = -1*np.dot(hk,dA_gk)
        phi0 = self.calcA_g(gk)
        dphi0 = np.inner(self.gradient_dA_dgi(dgk),dgk)
        a = 0
        b = alpha
        phib = self.phi(gk, dgk, b)
        l = 0
        while (phib <= phi0 + beta * )
        
        
    
    def inverseHessianPlusOne(self,delgk,deldA_gk,Hk):
        """
        

        Parameters
        ----------
        delgk : TYPE
            DESCRIPTION.
        deldA_gk : TYPE
            DESCRIPTION.
        Hk : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        s = delgk
        y = deldA_gk
        p = 1/(np.inner(y,s))
        
        #term1
        ip1 = (s - np.dot(Hk,y))
        op1 = np.outer(ip1,s)
        t1 = p*op1
        print('t1 ', t1)
        #term2
        ip2 = (s-np.dot(Hk,y))
        op2 = np.outer(s,ip2)
        t2 = p * op2
        print('t2 ', t2)
        
        #term3
        ip3 = np.inner((s-np.dot(Hk,y)),y)
        op3 = np.outer(s,s)
        t3 = (p**2)*ip3*op3
        print('t3 ', t3)
        
        Hk1 = Hk + t1 + t2 - t3
        
        
        
        return Hk1
    
    def oneIonWHAM_BFGS(self,bfgs_il,bfgs_tol,alpha,tao,beta,lsil,pcheck):
        """
        

        Parameters
        ----------
        bfgs_il : TYPE
            DESCRIPTION.
        bfgs_tol : TYPE
            DESCRIPTION.
        alpha : TYPE
            DESCRIPTION.
        tao : TYPE
            DESCRIPTION.
        beta : TYPE
            DESCRIPTION.
        lsil : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        q = 0
        (f0, gk) = self.initialEstimates()
        print('inital estimate = f0, g0: ', f0, gk)
        dA_gk = self.gradient_dA_dgi(gk)
        Hk = np.identity(np.size(dA_gk))
        while (q < bfgs_il): # and tol < bfgs_tol
            #line search
            gk1 = self.armijo_LineSearch(gk,dA_gk,Hk,alpha,tao,beta,lsil)
            #BFGS update
            dA_gk1 = self.gradient_dA_dgi(gk1)
            delgk = gk1 - gk
            deldA_gk = dA_gk1 - dA_gk
            Hk1 = self.inverseHessianPlusOne(delgk, deldA_gk, Hk)
            #update iterative terms
            gk = gk1
            dA_gk = dA_gk1
            Hk = Hk1
            q += 1
            print('iteration ', q)
            print('gk',gk)
            print('dA_gk', dA_gk)
            print('hk', Hk)
            if(q%pcheck == 0):
                f_gk = np.exp(gk)
                print('fgk: ', f_gk)
                rho = np.zeros(np.size(self.hist))
                for l in self.binI:
                    sum1 = 0
                    for i in self.simI:
                        sum1 += self.N_i[i]*f_gk[i]*self.c_i_l[i][l]
                    rho[l] = self.hist[l]/sum1
                print(rho)
                x = self.binI
                y = rho
                
                # plot
                fig, ax = plt.subplots()
                
                ax.plot(x, y, linewidth=2.0)
                
                plt.show()
            
        
            
                    
                    
        
        
            
            
            
            
            
            
            
            
            
            
        
        
        
        
        
# %% Test cell

xmin = 0
xmax = 3
bStepNum = 90
N = 5000000
k = 36
binSize = 0.0005
fname = 'data-files/xmn0_xmx3_bStepNum90_binSize0.0005_k36_N5000000_smoothFunc.txt'

test = WhamConvergence_1Ion(xmin, xmax, bStepNum, N, binSize, k, fname)

print(test.N_i)

test.oneIonWHAM_BFGS(200,0.01,1,0.9,0.5,500,2)

# xmin = 0
# xmax = 3
# bStepNum = 100
# N = 50000000
# k = 9
# binSize = 0.001
# test.oneIonWHAM_BFGS(200,0.01,1,0.9,0.1,500,2)
        
        
        
        
        