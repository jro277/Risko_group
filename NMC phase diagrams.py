# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:46:51 2018

@author: jro277
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

#Where h = Planck, kj = Boltzman in J, kev = Boltzman in eV
#T = list of temps in K, S = list of species, Sb = S with 'Bare', M = dict of species masses in kg
#P = list of pressures in pascals, X = list of crystal planes
#ep = dict of species epsilon values
#Area = dict of crystal plane surface areas in sq. Angs
#SNv = 2D DataFrame, the Nx values for each mu for a given species in expt
#Enmc = E in eV for one formula unit of bulk NMC
#BareG = dict of bare surface energies for each x
#Nnmc = dict setting Nnmc to 6 or 12 based on X
#x- and yrange = the list of all coords to plug into the graph

#L = 2D Dataframe, cube of de Broglie thermal wavelength for each species at each temp
#cf,u = multi-index column names
#CF = 3D Dataframe, the correction factor for each species, temp, and pressure
#SE = 2D Dataframe, DFT ground-state binding energies of species preferably bound to crystal planes
#MU = 4D Dataframe, chemical potentials for each species on each crystal plane at each temp and pressure, deprecated
#MP = Minimum energy Phase; 2D Dataframe, the species which can appear on the
#     phase diagram for the given T and X.
#cdict = dict mapping species to a number for matplotlib, maybe useless?
#Gn = 5D DataFrames, gamma based on P(H2, O2, H2O), T, and S; collated by Gamma = a dict to map X and G
#CV = a list of all tuples that can be used to specify a coordinate in Gamma-space
#sname1 and 2 = filler names to help make the math easier to read
#sites = dict specifying site names for each X
#biS, biSb, bidata, etc. = versions of S, Sb, Data, etc for bi-molecular adsorbates
#oldE, oldBareG, oldData = older versions, now obsolete due to fixed LDA+U calcs

h = 6.62607004E-34
kj = 1.38064852E-23
kev = 8.6173303E-5
T = [300, 600, 900, 1200]
S = ['H', 'H2', 'O', 'O2', 'OH', 'H2O']
Sb = ['H', 'H2', 'O', 'O2', 'OH', 'H2O', 'Bare']
biS = ['oo','oh','ow','hh','hw','ww']
biSb = ['oo','oh','ow','hh','hw','ww','Bare']
M = {'H':1.6737E-27, 'H2':3.3474E-27, 'O':2.65676E-26, 'O2':5.31352E-26, 'OH':2.82413E-26, 'H2O':2.9915E-26}
biM = {'oo':M['O2']*2, 'oh':M['O2']+M['OH'], 'ow':M['O2']+M['H2O'],
       'hh':M['OH']*2, 'hw':M['OH']+M['H2O'], 'ww':M['H2O']*2}
ep = {'H':-0.0043909,'H2':-6.6960279,'O':-0.00789,'O2':-8.7310591,
      'OH':7.0742182,'H2O':-14.222646, 'NMC':-182.68383853}
biep = {'oo':ep['O2']*2, 'oh':ep['O2']+ep['OH'], 'ow':ep['O2']+ep['H2O'],
        'hh':ep['OH']*2, 'hw':ep['OH']+ep['H2O'], 'ww':ep['H2O']*2}
P = [1*10**m for m in range(-14,13)]
P.sort()
X = ['n001', 'n012', 'n110', 'n218', 'n302']
Area = {'n001':54.317, 'n012':96.3, 'n110':137.73, 'n218':85.74883491, 'n302':49.35202118}
cdict = {a:S.index(a) for a in S}
SNv = pd.DataFrame([[-1.5,-1,2],[-3,-2,4],[-1,0,1],[-2,0,2],[-2.5,-1,3],[-4,-2,5],[0,0,0]],
                   index=Sb, columns=['NH2', 'NO2', 'NH2O'])
biSNv = pd.DataFrame([[-4,0,4],[-4.5,-1,5],[-3, -0.5, 4],[-2,-0.5,3],[-6.5,-3,8],[-8,-4,10],[0,0,0]],
                     index=biSb, columns=['NH2', 'NO2', 'NH2O'])
myEnmc = -60.82536319 #using my cell
oldEnmc = -60.894613 
Enmc = -60.96059656 #using garcia's cell
oldBareG = {'n001':-364.7597912, 'n012':-392.8123, 'n110':-794.11616}
BareG = {'n001':-352.8946149, 'n012':-338.7449697, 'n110':-690.1930954,
         'n218':-284.578906, 'n302':-229.1526252}
Nnmc = {'n001':6, 'n012':6, 'n110':12, 'n218':5, 'n302':4}
xrange = [x for x in np.linspace(-14, -6, num=100)]
yrange = [y for y in np.linspace(-17, -9, num=100)]
sites = {'n001':['s1','s2','s3','s4'], 'n012':['s1','s2','s3'], \
         'n110':['s1a','s2a','s3a','s4a','s5a','s5b','s6a','s6b'],\
             'n218':['s1','s2','s3','s4','s5','s6','s7','s8','s9'],
             'n302':['s1','s2','s3','s4','s5','s6','s7','s8','s9']}


def make_new_gamma(x, t, w, h2, o2):
    'Makes gamma, one line at a time, then returns the minimum value.'
    Gamma = pd.Series(index=Sb)
    L3 = (h/np.sqrt(2*np.pi*M['H2O']*kj*t))**3
    CF = kev*t*np.log(w*L3/(kj*t))
    for s in S:
        sname2 = Enmc*Nnmc[x] + np.dot([h2, o2, ep['H2O']+CF], SNv.loc[s,:])
        sname1 = (SE.loc[x,s]-(sname2))/Area[x]
        Gamma[s] = sname1
    Gamma['Bare'] = (BareG[x] - Enmc*Nnmc[x])/Area[x]
    b = Gamma.idxmin()
    return b

def make_bi_gamma(x,t,w,h2,o2):
    'Makes bi-molecular gamma, one line at a time, then returns the minimum value.'
    biGamma = pd.Series(index=biSb)
    L3 = (h/np.sqrt(2*np.pi*M['H2O']*kj*t))**3
    biCF = kev*t*np.log(w*L3/(kj*t))
    for s in biS:
        sname2 = Enmc*Nnmc[x] + np.dot([h2, o2, ep['H2O']+biCF], biSNv.loc[s,:])
        sname1 = (biSE.loc[x,s]-(sname2))/Area[x]
        biGamma[s] = sname1
    biGamma['Bare'] = (BareG[x] - Enmc*Nnmc[x])/Area[x]
    b = biGamma.idxmin()
    return b

def exptl(t):
    'Draws a box representing experimental pressures'
    c = []
    d = {'H2':xrange, 'O2':yrange}
    for s in ['H2', 'O2']:
        for p in [1e-10, 1e5]:
            L3 = (h/np.sqrt(2*np.pi*M[s]*kj*t))**3
            CF = kev*t*np.log(p*L3/(kj*t))
            MU = ep[s]+CF
            for i in d[s]:
                if MU <= i:
                    c.append(d[s].index(i))
                    break
                else: pass
    verts = [(c[0],c[2]),(c[0],c[3]),(c[1],c[3]),(c[1],c[2]),(c[0],c[2])]
    codes=[matplotlib.path.Path.MOVETO, matplotlib.path.Path.LINETO, matplotlib.path.Path.LINETO,\
           matplotlib.path.Path.LINETO, matplotlib.path.Path.CLOSEPOLY]
    path = matplotlib.path.Path(verts, codes)
    patch = matplotlib.patches.PathPatch(path, facecolor='none', lw=1)
    return patch

def colors(x, SL, b):
    '''Maps species and site to a color:
    O2,hh = Dark Green, H2O,ww = Teal, Bare = Cream. 
    H,oo = Green, O,ow = Magenta, H2,oh = Yellow, OH,hw = Blue.
    '''
    scd = {'H':[0,255,0], 'H2':[255,255,0], 'O':[255,0,255], 'O2':[11,82,46],
           'OH':[0,0,255], 'H2O':[82,201,211], 'Bare':[238,238,238],
           'oo':[0,255,0], 'oh':[255,255,0], 'ow':[255,0,255], 'hh':[11,82,46],
           'hw':[0,0,255], 'ww':[82,201,211]}
    if b == 'Bare':
        color = scd[b]
    else:
        bs = b.split('s')
        color = scd[bs[0]]
    return color

#uni-molecular adsorption
L = pd.DataFrame(index=T, columns=S)
for t in T:
    for s in S:
        L.loc[t,s] = (h/np.sqrt(2*np.pi*M[s]*kj*t))**3
cf = pd.MultiIndex.from_product([T,S])
CF = pd.DataFrame(index=P, columns=cf)
for i in P:
    for j in cf:
        CF.loc[i,j] = kev*j[0]*np.log(i*L.loc[j[0],j[1]]/(kj*j[0]))
olddata = [[-364.9656603, -366.6108044, -370.5203494, -376.5524559, -368.9678376, -373.5197118],\
        [-393.9395626, -399.6338797, -136.6162476, -416.0397935, -402.2124758, -407.3844512],\
        [-787.8265786, -810.5243064, -302.1280531, -287.8490562, -814.1992213, -842.221]]
data = [[-354.1378904, -357.7868478, -359.4081165, -361.6130464, -361.388489, -366.6473135],\
        [-345.6984318, -350.9940645, -348.1873763, -353.3710295, -353.2339324, -358.8898638],\
        [-694.8224055, -698.7720986, -697.3531396, -700.1243248, -702.6102063, -733.602346],\
        [-290.9674295, -291.9164854, -290.7771469, -294.4063503, -296.0676028, -300.1493118],
        [-234.3246807, -239.7791582, -237.3301264, -241.352055, -241.9195293, -247.8137567]]
SE = pd.DataFrame(data=data, index=X, columns=S)

#bi-molecular adsorption
biL = pd.DataFrame(index=T, columns=biS)
for t in T:
    for s in biS:
        biL.loc[t,s] = (h/np.sqrt(2*np.pi*biM[s]*kj*t))**3
bicf = pd.MultiIndex.from_product([T,biS])
biCF = pd.DataFrame(index=P, columns=bicf)
for i in P:
    for j in bicf:
        biCF.loc[i,j] = kev*j[0]*np.log(i*biL.loc[j[0],j[1]]/(kj*j[0]))
bidata = [[-370.3362474, -370.1683527, -375.4291002, -369.1464952, -374.5991898, -379.576143],
          [-362.0147225, -362.0974023, -367.2435075, -361.0071415, -367.9428559, -373.1864022],
          [-709.8024656, -711.9847003, -715.8975521, -714.0257368, -717.8218626, -721.2939495],
          [-304.6970616, -305.7961112, -310.2513945, -306.6737289, -310.8994213, -314.9520683],
          [-251.7379605, -251.4545318, -257.2805436, -252.3772306, -258.1883764, -263.0123848]]
biSE = pd.DataFrame(data=bidata, index=X, columns=biS)



def Phase_Diagram(m):
    'm = "uni" or "bi" for unimolecular or bimolecular data, respectively.'
    for x in X: #each x needs to be its own set of graphs, since each depends on (t,p,p,p)
        fig, ax = plt.subplots(4,4, gridspec_kw={'wspace':0.25}, figsize=(22,22)) #set up graphs
        pt = [12,38,63,88]; xtl = ['-13', '-11', '-9', '-7']; ytl = ['-16', '-14', '-12', '-10']
        if m == 'uni':
            SL = [a+b for a in S for b in sites[x]]
        elif m == 'bi':
            SL = [a+b for a in biS for b in sites[x]]
        else:
            print('Error: please specify "uni" or "bi" in Phase_Diagram\'s arguments.')
            return
        SL.append('Bare')
        WP = [1e3, 1e-1, 1e-5, 1e-9]; wp = [r'$10^1$', r'$10^{-3}$', r'$10^{-7}$', r'$10^{-11}$']
        for i in range(4): #rows, T
            for j in range(4): #columns, P(H2O)
                test = np.zeros((len(yrange),len(xrange),3), dtype='int')
                for h2 in xrange:
                    for o2 in yrange:
                        if m == 'uni':
                            b = make_new_gamma(x,T[i],WP[j],h2,o2)
                        else:
                            b = make_bi_gamma(x,T[i],WP[j],h2,o2)
                        test[yrange.index(o2),xrange.index(h2),:] = colors(x, SL, b)
                ax[i,j].imshow(test,cmap='Set1',vmin=0, vmax=8, origin='lower')
                if j == 0:
                    ax[i,j].set_yticks(pt); ax[i,j].set_yticklabels(ytl, fontsize=28)
                else:
                    ax[i,j].set_yticks([]); ax[i,j].set_yticklabels([])
                if i == 3:
                    ax[i,j].set_xticks(pt); ax[i,j].set_xticklabels(xtl, fontsize=28)
                else:
                    ax[i,j].set_xticks([]); ax[i,j].set_xticklabels([])
                ax[0,j].set_title(wp[j], fontsize=30)
                ax[3,j].set_xlabel(r'$\mu_{H_2}$ (eV)', fontsize=30)
                ax[i,0].set_ylabel(r'$\mu_{O_2}$ (eV)', fontsize=30)
                ax[i,j].add_patch(exptl(T[i]))
            ax[i,3].text(110, 50, str(T[i])+' K', ha='left', va='center', rotation='vertical', fontsize=30)
        fig.suptitle(r'$P_{H_2O}$ (mbar)', fontsize=30, y=0.93)
        if m == 'uni':
            fig.savefig('%s Unimolecular Phase Diagram-z.png' % x)
        else:
            fig.savefig('%s Bimolecular Phase Diagram-z.png' % x)
    return

