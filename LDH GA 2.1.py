# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:08:17 2019

@author: jro277
"""

import numpy as np
import pymatgen as mg
import pymatgen.analysis.diffraction.xrd as xrd
import pandas as pd
import itertools as its
from collections import namedtuple
import matplotlib
import matplotlib.pyplot as plt

#Define constants and tools
poss = range(0, 72)
xy_poss = its.product(np.linspace(0.0,1.0,num=12,endpoint=False),\
                            np.linspace(0.0,1.0,num=6,endpoint=False), repeat=1)
#z_poss = list(np.linspace(0.0, 1.0, num=24, endpoint=False))
xy_pos_dict = dict(zip(poss, xy_poss))
#z_pos_dict = dict(zip(range(0,24), z_poss))
A_pos = [0, 16, 26, 36, 52, 62]
B_pos = [2, 12, 28, 38, 48, 64]
#C_pos = [4, 14, 24, 40, 50, 60] #reference only, not actually used
#poly_dict = {'A':A_pos, 'B':B_pos, 'C':C_pos}
#laer_z = {0:4, 1:12, 2:20}
ra = np.random.randint
#expt is Dr. Song's exptl data by peak. 1st = 2-theta, 2nd = FWHM, 3rd = Intensity
expt = np.array([[11.6093069, 0.33422539, 1],
                 [20.2156104, 0.45120426, 0.230390492],
                 [23.3406177, 0.42613735, 0.468081494],
                 [35.472999, 0.3342209, 0.269779287],
                 [36.1080272, 0.2172465, 0.396604414],
                 [40.5365135, 0.434493, 0.223938879],
                 [47.7724929, 0.52640495, 0.226655348],
                 [63.1969942, 0.3175141, 0.130730051],
                 [64.4336281, 0.4679155, 0.106112054]])
xrds = xrd.XRDCalculator()
generation = 0
dna_index = {}
#Profac is Ye, Z's Profile Factor which compares calc and exptl xrd intensities
#Profac is not currently used in the fitness calculation, similar to MPerr.
fitdex = pd.DataFrame(columns=['matchfit', 'matchtot', 'fitsum','Profac'])

#Define DNA as nested namedtuples
DNA = namedtuple('DNA', 'Polytope L1 L2 L3 Lattice')
POL = namedtuple('POL', 'Layers Symmetry')
LAY = namedtuple('LAY', 'N Mineral') #Mineral is b or g, might affect certain things about M+ layer
SYM = namedtuple('SYM', 'Cn, Sz, Sxy, Tx, Ty') #Sym elements affect M+ layer stacking
LN = namedtuple('LN', 'C W1 W2') 
M = namedtuple('M', 'pos phi theta')
#Sym contents: 
#Cn = number indicating n subscript, note a C2 = sx + sy
#Sz = 0 or 1, an xy plane running through metal atoms
#Sxy = (-1:1); -1 indicates a zx plane at b = 0.5; 1 for zy plane, 0 means no plane, both is a C2
#Txy = (x,y); a vector indicating x|a and y|b distances to translate each layer.

#Define genedex as pandas multi-indices and fitdex as DataFrame
def init_genedex(abc=True):
    'Creates the blank genedex as a multi-index. Pass abc = False to use xyz-lattice vectors.'
    iterables = [['Layer1', 'Layer2', 'Layer3'], ['Carbonate', 'Water1', 'Water2'],\
                 ['pos', 'phi', 'theta']]
    poly = [('Polytope', 'Layers', 'N'), ('Polytope', 'Layers', 'Mineral'),\
            ('Polytope','Symmetry','Cn'), ('Polytope','Symmetry','Sz'),\
            ('Polytope','Symmetry','Sxy'), ('Polytope','Symmetry','Tx'), ('Polytope','Symmetry','Ty')]
    xrdcols = [['XRD'], [''], ['NoMP', 'MPerr', 'TotalPs', 'MPstring','Profac']]
    mi1 = pd.MultiIndex.from_product(iterables)
    mi2 = pd.MultiIndex.from_tuples(poly)
    mi3 = pd.MultiIndex.from_product(xrdcols)
    df1 = pd.DataFrame(columns=mi1)
    df2 = pd.DataFrame(columns=mi2)
    df3 = pd.DataFrame(columns=mi3)
    if abc: laterables = [['Lattice'], ['Sides', 'Angles'], ['a', 'b', 'c']] #tests for old or new lattice
    else: laterables = [['Lattice'], ['A', 'B', 'C'], ['x', 'y', 'z']]
    mi4 = pd.MultiIndex.from_product(laterables)
    df4 = pd.DataFrame(columns=mi4)
    genedex = pd.concat([df2, df1, df4, df3], axis=1)
    return genedex

def init_children(abc=True):
    'Creates the children multi-index. Pass abc = False to use xyz-lattice vectors.'
    iterables = [['Layer1', 'Layer2', 'Layer3'], ['Carbonate', 'Water1', 'Water2'],\
                 ['pos', 'phi', 'theta']]
    poly = [('Polytope', 'Layers', 'N'), ('Polytope', 'Layers', 'Mineral'),\
            ('Polytope','Symmetry','Cn'), ('Polytope','Symmetry','Sz'),\
            ('Polytope','Symmetry','Sxy'), ('Polytope','Symmetry','Tx'), ('Polytope','Symmetry','Ty')]
    mi1 = pd.MultiIndex.from_product(iterables)
    mi2 = pd.MultiIndex.from_tuples(poly)
    df1 = pd.DataFrame(columns=mi1)
    df2 = pd.DataFrame(columns=mi2)
    if abc: laterables = [['Lattice'], ['Sides', 'Angles'], ['a', 'b', 'c']] #tests for old or new lattice
    else: laterables = [['Lattice'], ['A', 'B', 'C'], ['x', 'y', 'z']]
    mi4 = pd.MultiIndex.from_product(laterables)
    df4 = pd.DataFrame(columns=mi4)
    children = pd.concat([df2, df1, df4], axis=1)
    return children

def Ribosome(run, generation, abc=True, path=None):
    '''Read a csv index file and compile the genedex, given the run and generation.
    Pass abc = False to use xyz-lattice vectors.
    Pass a pathname to import a different sub-pop genedex that doesn't rely on run or generation or version.'''
    genedex = init_genedex(abc)
    if path == None:
        strindex = pd.read_csv('./GA2/Genetics%s/Genedex Gen %s.csv' % (run,generation), index_col=0)
    else:
        strindex = pd.read_csv(path, index_col=0)
    strindex = strindex[0:(len(strindex)-3)] #removes first two nan lines from bad MI conversion.
    strist = []
    strict = {}
    counter = 0
    for i in strindex: #needed to handle mixed dtypes
        if i == 'Polytope.1':
            strist.append(strindex[i])
        elif i == 'XRD.3':
            strist.append(strindex[i])
        else: 
            strist.append(pd.to_numeric(strindex[i]))
    for i in genedex:
        strict[counter] = i
        counter += 1
    for i in range(len(strindex.columns)):
        genedex[strict[i]] = strist[i]
    #genedex = genedex.convert_objects(convert_numeric=True) #deprecated
    return genedex

def Transcription(fn, genedex):
    'Reads an entry of the genedex and makes a DNA object.'
    rna = []
    for i in genedex.loc[fn, :]:
        rna.append(i)
    lay = LAY(rna[0], rna[1])
    sym = SYM(rna[2], rna[3], rna[4], rna[5], rna[6])
    poly = POL(lay, sym)
    c1 = M(rna[7], rna[8], rna[9])
    w11 = M(rna[10], rna[11], rna[12])
    w12 = M(rna[13], rna[14], rna[15])
    c2 = M(rna[16], rna[17], rna[18])
    w21 = M(rna[19], rna[20], rna[21])
    w22 = M(rna[22], rna[23], rna[24])
    c3 = M(rna[25], rna[26], rna[27])
    w31 = M(rna[28], rna[29], rna[30])
    w32 = M(rna[31], rna[32], rna[33])
    if ('Lattice', 'A', 'x') in genedex.columns: #tests for old or new lattice
        lattice = np.array([[rna[34],rna[35],rna[36]],\
                        [rna[37],rna[38],rna[39]],\
                        [rna[40],rna[41],rna[42]]])
    else:
        lattice = np.array([[rna[34], rna[35], rna[36]],\
                            [rna[37], rna[38], rna[39]]])
    laer=[]
    for i in [0,1,2]:
        clist = [c1, c2, c3]
        wl1 = [w11, w21, w31]
        wl2 = [w12, w22, w32]
        a = LN(clist[i], wl1[i], wl2[i])
        laer.append(a)
    global dna_index
    dna_index[fn] = DNA(poly, laer[0], laer[1], laer[2], lattice)
    return

def setlat(sn):
    '''Generates the crystal lattice for each layer by getting the thickness
    from the Mineral gene and the Ulibarri-91 reference.
    Then makes the crystallographic matrix, g.
    '''
    thick_dict = {'b': 4.77, 'g': 4.85} #thicknesses from Ulibarri('91) for brucite and gibbsite
    lat = [j for i in sn.Lattice for j in i]
    lat[2] = thick_dict[sn.Polytope.Layers.Mineral] #uses the Mineral gene to set M+ layer thickness
    #lat = mg.Lattice.from_lengths_and_angles(lat[0:3], lat[3:6]) #deprecated
    lat = mg.Lattice.from_parameters(lat[0], lat[1], lat[2], lat[3], lat[4], lat[5])
    g = np.array([[lat.a**2,lat.a*lat.b*np.cos(np.radians(lat.gamma)),lat.a*lat.c*np.cos(np.radians(lat.beta))],\
               [lat.a*lat.b*np.cos(np.radians(lat.gamma)),lat.b**2,lat.b*lat.c*np.cos(np.radians(lat.alpha))],\
               [lat.a*lat.c*np.cos(np.radians(lat.beta)),lat.b*lat.c*np.cos(np.radians(lat.alpha)),lat.c**2]])
    return lat, g

def Layer_1(sn):
    '''Builds the first M+ layer.
    This approach attempts to build good metal oxide octahedra given the lattice gene.
    This means that M-O bond lengths may not be as expected in a structure, but hopefully the
    GA corrects for that over time via the XRD fitness.'''
    #Li = [24, 60]
    #Al = [4, 14, 40, 50]
    m_pos = [24, 60, 4, 14, 40, 50]
    lat, g = setlat(sn)
    atoms = ['Li', 'Li', 'Al', 'Al', 'Al', 'Al',\
             'O', 'H', 'O', 'H', 'O', 'H', 'O', 'H', 'O', 'H', 'O', 'H',\
             'O', 'H', 'O', 'H', 'O', 'H', 'O', 'H', 'O', 'H', 'O', 'H']
    coords = []
    #M's first
    for i in range(6):
        x = xy_pos_dict[m_pos[i]][0]
        y = xy_pos_dict[m_pos[i]][1]
        z = 0.5
        coords.append([x,y,z])
    #OH's next
    od = {0:(A_pos, -1), 1:(B_pos, 1)}
    for i in range(12): #use of coord_transform not needed here
        x = xy_pos_dict[od[i//6][0][i%6]][0]
        y = xy_pos_dict[od[i//6][0][i%6]][1]
        p = np.sqrt((2/3*lat.b)**2-(lat.a/6)**2) #octahedral edge length
        p1 = np.sqrt((p**2)/2-(lat.a/6)**2)
        z = od[i//6][1]*p1/lat.c + 0.5 #get frac dist up or down from M
        coords.append([x,y,z])
        hx = x; hy = y #H in same xy as O
        hz = z + od[i//6][1]*0.96/lat.c #H is set distance away from O
        coords.append([hx,hy,hz])
    s = mg.IStructure(lat, atoms, coords)
    return s

def Layer_transform(s, sn):
    'This takes the previous M+ layer and transforms it according to the symmetry genes.'
    #define symmop as mg.symmop, then apply using s.apply(symop)
    lat = s.lattice; atoms = s.species; coords = s.frac_coords
    s1 = mg.Structure(lat, atoms, coords)
    i = sn.Polytope.Symmetry #rotate by Cn, translate by (x,y) vector
    if i.Cn == 0: Cn = 0
    else: Cn = 360/i.Cn
    so = mg.SymmOp.from_axis_angle_and_translation(axis=[0,0,1], angle=Cn,\
                                                   translation_vec=(i.Tx,i.Ty,0))
    s1.apply_operation(so, fractional=True)
    #reflect across xy plane
    if i.Sz == 1:
        so = mg.SymmOp.reflection([0,0,1], origin=(0.5,0.5,0.5))
        s1.apply_operation(so, fractional=True)
    else: pass
    #reflect across xz or yz plane
    if i.Sxy == 0: pass
    else:
        if i.Sxy < 0: #zx plane
            nd = (0,1)
        elif i.Sxy > 0: #zy plane
            nd = (1,0)
        so = mg.SymmOp.reflection([nd[0],nd[1],0],origin=(0.5,0.5,0.5))
        s1.apply_operation(so, fractional=True)
    lat = s1.lattice; atoms = s1.species; coords = s1.frac_coords
    snew = mg.IStructure(lat, atoms, coords, to_unit_cell=True)
    return snew

def Layer_A(sn, layer):
    'Generates A- layers.'
    atoms = ['C', 'O', 'O', 'O']
    lat, g = setlat(sn)
    coords = C_positions(sn, layer) #start the carbonate with C, then with O.
    oc, oa = O_positions(sn, layer, lat, g)
    for i in oc:
        coords.append(i)
    if len(oa): #only add water atoms if O_pos found water in layer
        for i in oa:
            atoms.append(i)
        hc, ha = H_positions(sn, layer, lat, g)
        for i in hc:
            coords.append(i)
        for i in ha:
            atoms.append(i)
    else: pass
    s = mg.Structure(lat, atoms, coords)
    return s

def coord_transform(g, a, b, c, l):
    'Checks bond lengths and stretches them if needed.'
    v = np.array([a,b,c])
    d = np.round(np.sqrt(np.matmul(v, np.matmul(g,v))),decimals=3)
    if d == l:
        new_coords = a,b,c
    else: #if distance is wrong, fix it
        r = l/d
        T = np.array([[r,0,0],[0,r,0],[0,0,r]])
        new_coords = np.matmul(T,v)
    return new_coords

def O_positions(sn, layer, lat, g):
    'Determines O positions from structure_name and writes to Layer_A.'
    ld = {1:sn.L1, 2:sn.L2, 3:sn.L3}
    coords = []; atoms = []; B = np.radians(lat.gamma)-np.pi/2 #correction factor
    l = 1.285 #C-O bond length
    for ox in range(0,3): #write carbonate oxygens
        xy = xy_pos_dict[ld[layer].C.pos]
        td = {0:[0,2*np.pi/3,4*np.pi/3], 1:[2*np.pi/12, 2*np.pi*5/12, 2*np.pi*9/12]} #theta
        pd = {0:np.pi/2, 1:np.radians(90-ld[layer].C.phi), 2:np.radians(90+ld[layer].C.phi)} #phi
        a = np.sin(pd[ox])*np.cos(td[ld[layer].C.theta][ox]-B)*l/lat.a
        b = np.sin(pd[ox])*np.sin(td[ld[layer].C.theta][ox])*l/lat.b
        c = np.cos(pd[ox])*l/lat.c
        x,y,z = coord_transform(g, a, b, c, l)
        x = x + xy[0]
        y = y + xy[1]
        z = z + 0.5
        coords.append((x,y,z))
    #write water oxygens
    wnum = ['W1', 'W2']
    if True in np.isnan(ld[layer].W1): #check for water in layer
        wnum.remove('W1') 
    if True in np.isnan(ld[layer].W2): wnum.remove('W2')
    if len(wnum) == 0: #no water in layer, skip
        return coords, atoms
    else:
        for j in wnum:
            w = {'W1': ld[layer].W1, 'W2':ld[layer].W2}
            z = 0.5
            try: xy = xy_pos_dict[w[j].pos]
            except: print(layer, j, w[j])
            x = xy[0]
            y = xy[1]
            coords.append((x,y,z))
            atoms.append('O')
        return coords, atoms

def H_positions(sn, layer, lat, g):
    'Determine H positions and write to structure.'
    ld = {1:sn.L1, 2:sn.L2, 3:sn.L3}
    coords = []; atoms = []
    l = 0.9578 #O-H bond length
    m = np.radians(104.4776/2) #length and angles from NIST
    n = np.radians(lat.gamma)-np.pi/2 #correction factor
    clocktuple = {0:(m, -m), 1:(m+n, -m+n), 2:(m+2*n, -m+2*n), 3:(m+3*n, -m+3*n)}
    #write water hydrogens
    wnum = ['W1', 'W2']
    if True in np.isnan(ld[layer].W1): wnum.remove('W1')
    if True in np.isnan(ld[layer].W2): wnum.remove('W2')
    if len(wnum) == 0: #no water in layer, skip
        return coords, atoms
    else:
        for j in wnum: #write first (and second) waters
            w = {'W1': ld[layer].W1, 'W2':ld[layer].W2} 
            ct = clocktuple[w[j].theta]
            phi = np.radians(90-w[j].phi)
            for th in ct: #write each hydrogen in ct pair
                xy = xy_pos_dict[w[j].pos]
                a = np.sin(phi)*np.cos(th-n)*l/lat.a
                b = np.sin(phi)*np.sin(th)*l/lat.b
                c = np.cos(phi)*l/lat.c
                x,y,z = coord_transform(g, a, b, c, l)
                x = x + xy[0]
                y = y + xy[1]
                z = z + 0.5
                coords.append((x,y,z))
                atoms.append('H')
        return coords, atoms

def C_positions(sn, layer):
    'Determine C positions.'
    ld = {1: sn.L1, 2:sn.L2, 3:sn.L3}
    z = 0.5
    xy = xy_pos_dict[ld[layer].C.pos]
    x = xy[0]; y = xy[1]
    coords = [(x,y,z)]
    return coords

def Stack_Layers(sn, *args):
    'Takes input structures and stacks them.'
    layers = [i for i in args]
    c = layers[0].lattice.c
    lat = sn.Lattice
    atoms = [j for i in layers for j in i.species]
    coords = [j for i in layers for j in i.frac_coords]
    nlay = sn.Polytope.Layers.N*4 #number of half-layer-slabs
    midpts = [i/nlay for i in range(int(nlay)) if i%2 == 1]
    count = 0
    for i in range(len(midpts)):
        zs = [j[2] for j in layers[i].frac_coords] #get existing z's and then new ones
        newzs = [(z-0.5)*(c)/lat[0][2]+midpts[i] for z in zs]
        for j in range(len(newzs)): #replace old z's with new z's.
            coords[count][2] = newzs[j]
            count += 1
    #lat = mg.Lattice.from_lengths_and_angles(lat[0], lat[1]) #deprecated
    lat = [j for i in sn.Lattice for j in i]
    lat = mg.Lattice.from_parameters(lat[0], lat[1], lat[2], lat[3], lat[4], lat[5])
    s = mg.IStructure(lat, atoms, coords)
    s1 = s.get_sorted_structure()
    return s1

def XRD_test(fn, s, genedex):
    '''Takes structure and makes XRD data, filters and analyzes, and adds the
    results to the genedex.'''
    a = xrds.get_pattern(s) #make xrd pattern
    c = np.stack((a.x, a.y), axis=-1) #combine into x,y coords
    c = c[c[:,1] > 10] #filter out peaks that are too small
    chklist=[]
    Ilist = []
    mpstring = 'x' #string of matching peaks
    for k,i in enumerate(c[:,0]): #for each peak in the calc xrd:
        x = 0 #exptl peak counter
        for j in expt[:,0]: #compare to each exptl peak:
            if np.abs(i-j) <= expt[x,1]: #if peak is closer than err
                chklist.append(np.abs(i-j)) #note the diff from expt
                mpstring += str(x)
                Ilist.append([x, c[k,1], expt[x,2]*100])
                break #move on to next calc peak
            else: #if peak is farther than err, check next exptl peak
                x += 1
                continue
    #After all calc peaks are checked:
    chkry = np.array(chklist)
    d = chkry.sum(0)/len(chkry)
    Isum = []
    for i in Ilist:
        Isum.append(np.max([i[1],i[2]])*np.log10(i[1]/i[2])**2)
    genedex.loc[fn, ('XRD', '', 'NoMP')] = len(set(mpstring))-1 #index Number of Matching Peaks
    genedex.loc[fn, ('XRD', '', 'MPerr')] = d #index Matching Peak error
    genedex.loc[fn, ('XRD', '', 'TotalPs')] = len(c) #index Total Peaks
    genedex.loc[fn, ('XRD', '', 'MPstring')] = mpstring
    genedex.loc[fn, ('XRD', '', 'Profac')] = np.sum(Isum)
    return genedex

def Fitness(fn, genedex, fitdex):
    'Reads XRD-data from the genedex to determine fitness and adds it to fitdex.'
    a = genedex.loc[fn, ('XRD', '', 'NoMP')]
    b = genedex.loc[fn, ('XRD', '', 'TotalPs')]
    if b > 9: b = b-((b-9)*2) #express TP fitness as how close to 9 TPs.
    c = a + b #fitsum is sum of fitness things
    d = genedex.loc[fn, ('XRD','','Profac')]
    #Check for collisions, if found, set fit parameters to 0 to limit reproduction
    for i in ['Layer1', 'Layer2', 'Layer3']:
        q = []
        for j in ['Carbonate', 'Water1', 'Water2']:
            if np.isnan(genedex.loc[fn, (i, j, 'pos')]): pass
            else: q.append(genedex.loc[fn, (i, j, 'pos')])
        qs = set(q)
        if len(qs) == len(q): pass
        else: 
            a = b = c = 0
            break
    fitdex.loc[fn, :] = [a,b,c,d]
    fitdex = fitdex.astype('float64')
    return fitdex

def Selection(fitdex, run, generation):
    'Sort and select the best and the randomly chosen extras for breeding.'
    sel = 0.36 #The best are chosen
    breeders = fitdex.nlargest(int(sel*len(fitdex)), 'fitsum')
    breeders.to_csv('./GA2/Genetics%s/Generation %s Breeders.csv' % (run, generation))
    a = len(breeders)*1.25 #add in some lucky extras for diversity
    while len(breeders) < a:
        breeders = pd.concat([breeders, fitdex.sample()])
    breedlist = list(breeders.index) #get list of fn's to pass into genedex
    for i in range(20): #duplicate best individual a bunch
        breedlist.append(breedlist[0])
    return breedlist

def Breed(breedlist): #Put Selection() in the args to pipe breedlist over
    'Mix up breeders and generate children. Assign the children to a genedex.'
    children = init_children(abc)
    kid = 0 #indexer
    for i in range(len(breedlist)): #p's chosen in order, m's at random
        p = dna_index[breedlist[i]]
        m = dna_index[breedlist[ra(len(breedlist))]]
        for j in range(np.random.choice([2,2,2,3])): #they have 2.25 kids
            children.loc[kid] = Child(p,m) #appends the child-list to the children-genedex
            kid += 1
    return children

def Child(p,m):
    'Given two parents, mix alleles to make the child. Whole anions are mixed, retaining their rotation.'
    inheritance = {0:p, 1:m} #coin-flipper
    child = [] #rna-esque list to hold nucleotides in order for the children-genedex
    layers = inheritance[ra(2)].Polytope.Layers; layers = Mutate(layers)
    sym = inheritance[ra(2)].Polytope.Symmetry; sym = Mutate(sym)
    for i in layers:
        child.append(i)
    for i in sym:
        child.append(i)
    lattice = inheritance[ra(2)].Lattice; lattice = Mutate(lattice)
    m = [inheritance[ra(2)].L1.C, inheritance[ra(2)].L1.W1, inheritance[ra(2)].L1.W2,\
         inheritance[ra(2)].L2.C, inheritance[ra(2)].L2.W1, inheritance[ra(2)].L2.W2,\
         inheritance[ra(2)].L3.C, inheritance[ra(2)].L3.W1, inheritance[ra(2)].L3.W2]
    for i in range(9):
        m[i] = Mutate(m[i]) #in- and output named tuple
        for j in m[i]:
            child.append(j) #add nucleotides in order
    for i in lattice: #lattice is an ndarray, so two slices needed to get nucs.
        for j in i:
            child.append(j)
    return child

def Mutate(gene):
    'Using a variable mutation chance, test and mutate the incoming genes.'
    s = np.random.random #random seed alias
    if type(gene) == np.ndarray: #mutate lattice paramaters with independent 1% chances
        if gene.size == 9: #test for old or new lattice
            mut = np.random.random() * 0.2 - 0.1 #keeps change small
            if s() >= 0.99: #a uses x only
                gene[0][0] += mut
            if s() > 0.99: #b needs trig to shift coord axes in x and y
                r = np.sqrt((gene[1][0])**2 + (gene[1][1])**2)
                r += mut
                gene[1][0] += r*-0.5
                gene[1][1] += r*np.sqrt(3)/2
            if s() > 0.99: #c uses z only
                gene[2][2] += mut
            else: pass
        else:
            mut = s()*0.2-0.1
            for i in range(6):
                if s() >= 0.98:
                    gene[i//3][i%3] += mut
                else: pass
    elif len(gene) == 2: #mutate polytope layers 
        n, m = gene
        if s() >= 0.99: #1% to mutate N
            a = gene.N - 1
            if s() >= 0.5: a += 1
            else: a -= 1
            a = a%3
            n = a + 1
        if s() >= 0.98: #2% to mutate Mineral
            mut = ['b', 'g']
            mut.remove(gene.Mineral)
            m = mut[0]
        gene = LAY(n,m)
    elif len(gene) == 5: #mutate polytope symmetry
        cn, sz, sxy, tx, ty = gene
        if s() >= 0.99: #Cn
            cn = gene.Cn
            if s() >= 0.5: cn += 1
            else: cn -= 1
            if cn == 13: cn = 2 #keep Cn in 1-12 range
            elif cn == 0: cn = 12
            elif cn < 0: cn = cn * -1
        if s() >= 0.98: #Sz
            mut = [0,1]
            mut.remove(gene.Sz)
            sz = mut[0]
        if s() >= 0.99: #Sxy
            mut = [-1, 0, 1, 2] #2 is having both Sx and Sy
            mut.remove(gene.Sxy)
            sxy = np.random.choice(mut)
            if sxy == 2: #this is equivalent to adding a C2 to the Cn
                sxy = 0
                try: 
                    a = 1/cn + 1/2
                    b = int(np.round(1/a))
                    cn = b
                except:
                    cn = 2
            else: pass
        if s() >= 0.98: #Txy
            a = s() * 0.5 - 0.25
            bx = np.sqrt(0.25**2 - a**2)
            b = s() * (2*bx) - bx
            tx = gene.Tx + a
            if tx < 0: tx += 1 #keep vectors within bounds of unit cell
            elif tx > 1: tx -= 1
            ty = gene.Ty + b
            if ty < 0: ty += 1
            elif ty > 1: ty -= 1
        gene = SYM(cn, sz, sxy, tx, ty)
    else: #mutate molecular tuple
        mut = [1, -1, 3, -3] #this should be changed to reflect new grid
        if s() > 0.95: #consider adding theta mutations
            a,b,c = gene
            b += np.random.choice(mut)
            if b > 30: b = 30
            elif b < -30: b = -30 #keep phi within 30 deg.
            if s() > 0.99:
                a += np.random.choice(mut)
                a = a % 72 #limit pos to valid numbers
            gene = M(a, b, c)
        else: pass
    return gene

this error intentionally left in to remind me where I am.
#might, might go back to check on H2O angles

#Main program:
run = input('Run: ')
abc = True
if generation == 0:
    genedex = Ribosome(run, generation, abc) #1st run pulls from the starting set
while generation <= 50:
    for fn in genedex.index:
        Transcription(fn, genedex) #Make DNA
        sn = dna_index[fn]
        s1 = Layer_1(sn)
        a1 = Layer_A(sn, 1)
        if sn.Polytope.Layers.N == 2: #build and stack additional layers if needed
            s2 = Layer_transform(s1, sn)
            a2 = Layer_A(sn, 2)
            s = Stack_Layers(sn, s1, a1, s2, a2)
        elif sn.Polytope.Layers.N == 3:
            s2 = Layer_transform(s1, sn)
            a2 = Layer_A(sn, 2)
            s3 = Layer_transform(s2, sn)
            a3 = Layer_A(sn, 3)
            s = Stack_Layers(sn, s1, a1, s2, a2, s3, a3)
        else:
            s = Stack_Layers(sn, s1, a1)
        XRD_test(fn, s, genedex) #test structure
        fitdex = Fitness(fn, genedex, fitdex) #Determine fitness
    children = Breed(Selection(fitdex, run, generation)) #Reproduction
    genedex.to_csv('./GA2/Genetics%s/Genedex Gen %s.csv' % (run,generation))
    genedex = children #Generation wrap-up and increment
    dna_index = {}
    fitdex = pd.DataFrame(columns=['matchfit', 'matchtot', 'fitsum'])
    generation += 1

#this is code to save a picture of the xrd spectrum
xrds.get_plot(s, annotate_peaks=False).savefig('./Pics/LDH Sample XRD.png')

