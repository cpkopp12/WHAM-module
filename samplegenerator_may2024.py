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


class SampleGenerator_OneIon:
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
        
        
        
        
        
# %% Test Cell
test = SampleGenerator_OneIon(0,3,30,10000,2,0.01)  
(a,b)=integrate.quad(test.smoothFunc,0,3)
print(a*test.normConstant(test.smoothFunc,0,3))
        

        