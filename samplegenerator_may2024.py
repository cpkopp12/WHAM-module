# -*- coding: utf-8 -*-
"""
author: Cameron Kopp

created: 5/13/2024
"""

# IMPORTS -----------------------------------------------
import numpy as np
import scipy.integrate as integrate
from numpy import random as r
from math import *
import matplotlib.pyplot as plt
from matplotlib import colors


class DataGenerator_OneIon:
    """
    Generate a data set which replicates a steered molecular dynamics 
        simulation through a given PMF lanscape for one ion pulled
        by a gaussian biasing potential 
    
    inputs: xmin, xmax (such that the data set will span the range xmin-max);
        baisingStepNumber (number of equally spaced centers for the biasing 
                           potential spaning xmin - xmax, not including 
                           xmin, xmax);
        sampleNumber (total number of samples included in the data set over 
                      all biasing steps);
        springConstant (biasing potential is that of a spring 
                        U=(1/2)k(x-xcenter)^2, springConstant is k, will
                        determine the standard diviation of samples around
                        the center of the biasing potential);
        binSize (samples will be place in bins along the range xmin to xmax,
                 binSize is width of these bins, such that a smaller binsize
                 will result in a greater number of bins)
        
    """
    def __init__(self, xmin, xmax, baisingStepNumber, sampleNumber, 
                 springConstant, binSize):
        """
    
        Parameters
        ----------
        xmin : TYPE
            DESCRIPTION.
        xmax : TYPE
            DESCRIPTION.
        baisingStepNumber : TYPE
            DESCRIPTION.
        sampleNumber : TYPE
            DESCRIPTION.
        springConstant : TYPE
            DESCRIPTION.
        binSize : TYPE
            DESCRIPTION.

        Takes inputs and initializes corresponding the histogram grid,
        creates biasing step indexes and coordinates, creates bin coordinates
        and indexes

        """
        self.xmn = xmin
        self.xmx = xmax
        self.bStepNum = baisingStepNumber
        self.N = sampleNumber
        self.k = springConstant
        self.binSize = binSize
        
        #set up two arrays, bin coordinates (center) and bin indicies
        binNumber = (self.xmx-self.xmn)/self.binSize
        #array with bin indexes = binI
        self.binI = np.arange(0,binNumber, dtype=int)
        #array with bin x coordinate values = binX
        self.binX = (self.binI * self.binSize) + (self.binSize/2)
        
        #set up two arrays, coordinates of the center of biasing potential
        #for each step biasX, and the indicies for the biasing steps baisI
        stepSize = (self.xmx-self.xmn)/(self.bStepNum + 1)
        self.biasI = np.arange(0, self.bStepNum, dtype=int)
        self.biasX = (self.biasI * stepSize) + stepSize
        
        #histogram array
        self.hist = np.zeros(np.size(self.binI))
        
        
        
    def biasPotential(self, x, mu):
        """
        

        Parameters
        ----------
        x : float
            value along x-coordinate
        mu : float
            center of biasing potential along x-coordinate

        Returns
        -------
        Value of a biasing potential center centered at x=mu for a given value 
        of x, max value of 1

        """
        
        #define constant
        c1 = self.k/2
        
        return np.exp(-c1*((x-mu)**2))
    
    
    
    def smoothFunc(self, x):
        """
        

        Parameters
        ----------
        x : float
            given value of x along x coordinate

        Returns
        -------
        value of the probabiltiy distribution function 
        rho(x) =
            (20/21)*((((1/(1 + (1 - (x/2))^2))*((Sin[2 x]^2) + 2)/3 ) + 1/20)),
        which is follows rho(x) <= 1

        """
        return (20/21)*((((1/(1+(1-(x/2))**2))*((np.sin(2*x)**2)+2)/3)+1/20))
        
        
    def normConstant(self, func, xmin, xmax):
        """
        calculate the normalization constant for probabilty function (func) 
        on xmin to xmax
        """
        
        (a,b) = integrate.quad(func, xmin, xmax)
        
        return 1/a
    
    def normalDistribution(self, x, mu, sig):
        
        c = 1/(sig*np.sqrt(2*np.pi))
               
        return c*np.exp(-(1/2)*(((x-mu)**2)/(sig**2)))
        
    
    def generateSample(self, rho):
        """
        
        Parameters: rho (function(x), distribution function, can be normalized
                         or unormalized)

        Returns
        -------
        self.hist, histogram of sample

        """
        
        #calculate norm constant of rho
        A = self.normConstant(rho,self.xmn,self.xmx)
        #calculate standard deviation for normal distribution from k
        sig = 1/np.sqrt(self.k)
        #number of samples per biasing step
        n = floor(self.N/self.bStepNum)
        
        #loop over biasing potential steps
        for mu in self.biasX:
            print(mu)
            #accepted sample counter i
            i = 0
            
            #loop over accept-reject algorithm while i < n
            while(i < n):
                #test sample value
                xt = r.normal(mu,sig)
                #value of biasingPotential(x) * normalized rho(x)
                f_xt = A*self.biasPotential(xt, mu)*rho(xt) 
                #value of normal distribuition(xt)
                n_xt = self.normalDistribution(xt, mu, sig)
                
                y = r.uniform(0,1)
                
                if y <= (f_xt/n_xt):
                    if (xt < self.xmx) and (xt > self.xmn):
                        i = i + 1
                        
                        histbinf = (xt/self.binSize)
                        histbin = floor(histbinf)
                        #hi is the histogram index that xt corresponds to
                        hi = int(histbin)
                        self.hist[hi] = self.hist[hi] + 1
                        
                        if (i == n/2):
                            print('1/2 way')
                        
                        
       

        # make data
        x = self.binI
        y = self.hist
        
        # plot
        fig, ax = plt.subplots()
        
        ax.plot(x, y, linewidth=2.0)
        
        plt.show()
        
        return self.hist
    
    
    def writeToFile(self, funcUsed):
        filename = "data-files/xmn{}_xmx{}_bStepNum{}_binSize{}_k{}_N{}_{}.txt".format(self.xmn,self.xmx, self.bStepNum, self.binSize, 
                            self.k, self.N, funcUsed)
        
        with open(filename,'w') as myfile:
            for x in self.binI:
                myfile.write('{} '.format(self.hist[x]))
                
        myfile.close()
        
        return
                        
                
        
        
        
        
        
# %% Test Cell
xmin = 0
xmax = 3
bStepNum = 60
N = 10000000
k = 16
binSize = 0.001

test = DataGenerator_OneIon(xmin, xmax, bStepNum, N, k, binSize)  
testhist = test.generateSample(test.smoothFunc)
test.writeToFile('smoothFunc')
        

        