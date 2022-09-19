import os
import matplotlib.pyplot as plt
import os
import pandas as pd
from amftrack.util.sys import get_analysis_folders,get_time_plate_info_from_analysis
import numpy as np
import imageio
import os
import cv2
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

def wave(xt,c,lamb,K,x0):
    x=xt[0,:]
    t = xt[1,:]
    return(K*(1/(1+np.exp(lamb*(x0+x-c*t)))))
def dwave(xt,c,lamb,K,x0):
    x=xt[0,:]
    t = xt[1,:]
    return(K*(np.exp(lamb*(x0+x-c*t))/(1+np.exp(lamb*(x0+x-c*t)))**2))
def dS(t,lamb,C,t0):
    return(C*((np.exp(lamb*(t0-t))/(1+np.exp(lamb*(t0-t)))**2)))

def S(t,lamb,C,t0):
    return(C*(1/(1+np.exp(lamb*(t0-t)))))


def get_wave_fit(time_plate_info,plate,timesteps,lamb = -1,C =0.2):
    table = time_plate_info.loc[time_plate_info["Plate"]==plate]
    table = table.replace(np.nan,-1)
    ts = list(table['timestep'])
    table = table.set_index('timestep')
    ts.sort()
    dic = {}
    tot_t = list(table.index)
    tot_t.sort()
    timesteps = tot_t[10:80]
    # timesteps = [tot_t[0],tot_t[80]]

    ts = []
    xs = []
    ys = []
    for time in timesteps:
    #     ax.set_yscale("log")

        maxL = np.sqrt(1900)
        X = np.linspace(0,maxL,100)
        incr = 100
        def density(x):
            area = x**2
            index = int(area//incr)
            column = f"ring_density_incr-100_index-{index}"
            return(float(table[column][time]))
        xvalues = np.array([np.sqrt(100*i) for i in range(20)])
        yvalues = [density(x) for x in xvalues]
        xvalues = np.sqrt((xvalues**2+table["area_sep_comp"][0])/(np.pi/2))
        xvalues = list(xvalues)
        tvalues = [table['time_since_begin_h'][time] for x in xvalues]
        ts+=tvalues
        xs += xvalues
        ys += yvalues
    xt = np.array((xs,ts))
    popt_f,cov = curve_fit(wave, xt,ys,bounds = ([0,0,0,-np.inf],[np.inf,np.inf,np.inf,np.inf]),p0=[0.2,-lamb,C,0])
    popt_f,cov = curve_fit(wave, xt,ys,bounds = ([0,0,0,-np.inf],[np.inf,np.inf,np.inf,np.inf]),p0=[0.2]+list(popt_f[1:]))
    # popt_f[0]/=1.5

    popt_f
    residuals = ys- wave(xt, *popt_f)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ys-np.mean(ys))**2)
    r_squared_dens = 1 - (ss_res / ss_tot)
    ts = []
    xs = []
    ys = []
    for time in timesteps:
    #     ax.set_yscale("log")

        maxL = np.sqrt(1900)
        X = np.linspace(0,maxL,100)
        incr = 100
        def density(x):
            area = x**2
            index = int(area//incr)
            column = f"ring_active_tips_density_incr-100_index-{index}"
            return(float(table[column][time]))
        xvalues = np.array([np.sqrt(100*i) for i in range(20)])
        yvalues = [density(x) for x in xvalues]
        xvalues = np.sqrt((xvalues**2+table["area_sep_comp"][0])/(np.pi/2))
        xvalues = list(xvalues)
        tvalues = [table['time_since_begin_h'][time] for x in xvalues]
        ts+=tvalues
        xs += xvalues
        ys += yvalues
    xt = np.array((xs,ts))
    ys = np.array(ys)
    pos = np.where(ys>=0)[0]
    xt = xt[:,pos]
    ys = ys[pos]
    popt_f2,cov = curve_fit(dwave, xt,ys,bounds = ([0,0,0,-np.inf],[np.inf,np.inf,np.inf,np.inf]),p0=[0.2,popt_f[1],popt_f[2]/3000,popt_f[3]])
    residuals = ys- dwave(xt, *popt_f2)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ys-np.mean(ys))**2)
    r_squared_tips = 1 - (ss_res / ss_tot)
    return(popt_f,r_squared_dens,popt_f2,r_squared_tips)