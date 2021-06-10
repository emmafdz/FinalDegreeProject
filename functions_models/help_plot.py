#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:06:02 2018

@author: gprat
"""
import numpy as np
import matplotlib.colors as colors
import matplotlib as mpl
import matplotlib.cm as cmx	
import matplotlib.pyplot as plt


def colormap(cmap,ival,vmin,vmax):
    cm = plt.get_cmap(cmap) 
    cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
#    color=scalarMap.to_rgba(ival)
#    color_dash='grey'
    return scalarMap.to_rgba(ival)
    
    
    
def remove_axis(ax): 
    '''
    remove axis top and right
    '''

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Only show ticks on the left and bottom spines
#    ax.yaxis.set_ticks_position('left')
#    ax.xaxis.set_ticks_position('bottom')
#    
    return 0
    
def remove_axis_bottom(ax): 
    '''
    remove axis top and right
    '''

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Only show ticks on the left and bottom spines
#    ax.yaxis.set_ticks_position('left')
#    ax.xaxis.set_ticks_position('bottom')
#    
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.spines['bottom'].set_visible(False)
    return 0  
    


   
    
def plot_potential(ax,model,coef=[0,1.,2,0],xmin=-0.7,xmax=0.7,ymin=-0.5,ymax=0.5,color_left='darkred',color_right='darkcyan',axis_off=True,linewidth=3):
    if axis_off:
        ax.axis('off')
    if model=='PI' or 'DDM' in model:
        print('hola PI',model,model)
        print('xmin',xmin,xmax)
        xminus=np.linspace(xmin,0,2)
        xplus=np.linspace(0,xmax,2)
        yplus=coef[0]*xplus
        yminus=coef[0]*xminus
        ax.plot(xminus,yminus,'-',color=color_left,linewidth=linewidth)                   
        ax.plot(xplus,yplus,'-',color=color_right,linewidth=linewidth)
        x_abs=[xmin,xmin]
        x_abs2=[xmax,xmax]
        if model=='DDMA':
            y_abs=[yminus[0],-0.25]
            ax.plot(x_abs,y_abs,'-',color=color_left,linewidth=linewidth)
            y_abs=[yplus[-1],-0.25]
            ax.plot(x_abs2,y_abs,'-',color=color_right,linewidth=linewidth)
            
        if model=='DDMR':    
            y_abs=[yminus[0],0.25]
            ax.plot(x_abs,y_abs,'-',color=color_left,linewidth=linewidth)
            y_abs=[yplus[-1],0.25]
            ax.plot(x_abs2,y_abs,'-',color=color_right,linewidth=linewidth)  
    

    
    elif model=='DW':
        xminus=np.linspace(xmin,0,100)
        xplus=np.linspace(0,xmax,100)
        yplus=potential_DW(xplus,coef)
        yminus=potential_DW(xminus,coef)
        print('hola DW 2',model)
        ax.plot(xminus,yminus,'-',color=color_left,linewidth=linewidth)                   
        ax.plot(xplus,yplus,'-',color=color_right,linewidth=linewidth)        


    ax.set_xlim(xmin*1.2,xmax*1.2)
    ax.set_ylim(ymin*1.2,ymax*1.2)
    
    return 0

def potential_TW(x,coef):
    return coef[0]*x+coef[1]*x**2+coef[2]*x**4+coef[3]*x**6
    

def potential_DW(x,coef):
    return -coef[0]*x-(coef[1]/2.)*x**2+(coef[2]/4.)*x**4
    

def yticks(ax,yticks,yticklabels=None,fontsize=10):
    if yticklabels==None:
        yticklabels=yticks
        
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels,fontsize=fontsize ) 
    return 0


def xticks(ax,xticks,xticklabels=None,fontsize=10):
    if xticklabels==None:
        xticklabels=xticks
        
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels,fontsize=fontsize ) 
    return 0


