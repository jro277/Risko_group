# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:46:00 2018

@author: josiah.roberts
"""
#import modules
import numpy as np
import pandas as pd
import pymatgen as mg
import pymatgen.analysis.diffraction.xrd as xrd
import itertools as its
from collections import namedtuple

#define constants and tools.

#define the xy grid and the z-coords all as fractions of the structure-specific lattice
#most atoms and molecules sit in one of these positions
poss = range(0, 18)
xy_poss = its.product((0, 0.166666666, 0.333333333, 0.5, 0.666666666, 0.8333333333),\
                            (0, 0.333333333, 0.666666666), repeat=1)
z_poss = list(np.linspace(0.0, 1.0, num=24, endpoint=False))
xy_pos_dict = dict(zip(poss, xy_poss))
z_pos_dict = dict(zip(range(0,24), z_poss))
laer_z = {0:4, 1:12, 2:20}
#define the ABC layers in terms of xy grid positions, used for layer stacking
A_pos = [0,5,7,9,14,16]
B_pos = [1,3,8,10,12,17]
C_pos = [2,4,6,11,13,15]
poly_dict = {'A':A_pos, 'B':B_pos, 'C':C_pos}
ra = np.random.randint #these are aliases
xrds = xrd.XRDCalculator()
#this is the experimental data that determines fitness. 
expt = np.array([[11.6093069, 0.33422539],
                 [20.2156104, 0.45120426],
                 [23.3406177, 0.42613735],
                 [35.472999, 0.3342209],
                 [36.1080272, 0.2172465],
                 [40.5365135, 0.434493],
                 [47.7724929, 0.52640495],
                 [63.1969942, 0.3175141],
                 [64.4336281, 0.4679155]])
#define important global variables and lists
generation = 0
dna_index = {}
coords = []
run = input('Run: ') #this allows for multiple genealogies stored in different folders

#Define DNA as nested namedtuples
DNA = namedtuple('DNA', 'Polytope L1 L2 L3 Lattice')
LN = namedtuple('LN', 'C W1 W2')
M = namedtuple('M', 'pos phi theta')

#Define genedex as pandas multi-indices and fitdex as DataFrame
def init_genedex(abc=False):
    '''This defines the genedex as a pandas MultiIndex.
    The crystal lattice may be defined using xyz vectors (abc = False) or
    by using side lengths and angles (abc = True).
    '''
    iterables = [['Layer1', 'Layer2', 'Layer3'], ['Carbonate', 'Water1', 'Water2'],\
                 ['pos', 'phi', 'theta']]
    poly = [['Polytope'], [''], ['']]
    xrdcols = [['XRD'], [''], ['NoMP', 'MPerr', 'TotalPs', 'MPstring']]
    mi1 = pd.MultiIndex.from_product(iterables) #NoMP is the Number of uniquely Matching Peaks
    mi2 = pd.MultiIndex.from_product(poly)
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
global fitdex #fitdex stores fitness information.
fitdex = pd.DataFrame(columns=['matchfit', 'matchtot', 'fitsum'])

def init_children(abc=False):
    'Creates the children MultiIndex. Pass lattice information as with genedex.'
    iterables = [['Layer1', 'Layer2', 'Layer3'], ['Carbonate', 'Water1', 'Water2'],\
                 ['pos', 'phi', 'theta']]
    poly = [['Polytope'], [''], ['']]
    mi1 = pd.MultiIndex.from_product(iterables)
    mi2 = pd.MultiIndex.from_product(poly)
    df1 = pd.DataFrame(columns=mi1)
    df2 = pd.DataFrame(columns=mi2)
    if abc: laterables = [['Lattice'], ['Sides', 'Angles'], ['a', 'b', 'c']] #tests for old or new lattice
    else: laterables = [['Lattice'], ['A', 'B', 'C'], ['x', 'y', 'z']]
    mi4 = pd.MultiIndex.from_product(laterables)
    df4 = pd.DataFrame(columns=mi4)
    children = pd.concat([df2, df1, df4], axis=1)
    return children

#Define all functions used in main program

def Transcription(fn, genedex):
    'Reads an entry of the genedex and makes a DNA object.'
    rna = []
    for i in genedex.loc[fn, :]:
        rna.append(i)
    poly = rna[0]
    c1 = M(rna[1], rna[2], rna[3])
    w11 = M(rna[4], rna[5], rna[6])
    w12 = M(rna[7], rna[8], rna[9])
    c2 = M(rna[10], rna[11], rna[12])
    w21 = M(rna[13], rna[14], rna[15])
    w22 = M(rna[16], rna[17], rna[18])
    c3 = M(rna[19], rna[20], rna[21])
    w31 = M(rna[22], rna[23], rna[24])
    w32 = M(rna[25], rna[26], rna[27])
    if ('Lattice', 'A', 'x') in genedex.columns: #tests for lattice vectors or parameters
        lattice = np.array([[rna[28],rna[29],rna[30]],\
                        [rna[31],rna[32],rna[33]],\
                        [rna[34],rna[35],rna[36]]])
    else:
        lattice = np.array([[rna[28], rna[29], rna[30]],\
                            [rna[31], rna[32], rna[33]]])
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

def Fitness(fn, genedex):
    'Reads XRD-data from the genedex to determine fitness and adds it to fitdex.'
    global fitdex
    a = genedex.loc[fn, ('XRD', '', 'NoMP')]
    b = genedex.loc[fn, ('XRD', '', 'TotalPs')]
    if b > 9: b = b-((b-9)*2) #express TP fitness as how close to 9 TPs.
    c = a + b #fitsum is sum of fitness things
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
    fitdex.loc[fn, :] = [a,b,c]
    fitdex = fitdex.astype('float64')
    return fitdex

def Structure_start(sn, fn, coords): 
    'Write lattice, elements, coords to pymatgen structure; returns pymatgen structure.'
    if sn.Lattice.size == 6: #tests for lattice vectors or parameters
        lattice = mg.Lattice.from_parameters(sn.Lattice[0][0],sn.Lattice[0][1],sn.Lattice[0][2],\
                                         sn.Lattice[1][0],sn.Lattice[1][1],sn.Lattice[1][2])
    else:
        lattice = sn.Lattice
    wn = 6 #the (max) number of water molecules. This part handles missing waters.
    for i in [sn.L1, sn.L2, sn.L3]:
        for j in [i.W1, i.W2]:
            if True in np.isnan(j):
                wn -= 1
            else: pass
    #Writes the list of atoms by element, which must be in the same order as the list of coordinates.
    On = wn + 45 
    Hn = (wn * 2) + 36
    atoms = []
    for i in range(On):
        atoms.append('O')
    for i in range(Hn):
        atoms.append('H')
    for i in range(6):
        atoms.append('Li')
    for i in range(12):
        atoms.append('Al')
    for i in range(3):
        atoms.append('C')
    s = mg.Structure(lattice, atoms, coords) #makes the pymatgen structure
    return s

def Polyseq(sn):
    'Determine polytope sequence using fn.'
    if sn.Polytope == 1: #this is all about how layers can stack.
        polyseq = ['C', 'C', 'B', 'B', 'A', 'A']
    else:
        polyseq = ['C', 'B', 'A', 'C', 'B', 'A']
    return polyseq

def writeatom(a, b, c):
    'Add frac_coords abc to the current pymatgen structure coords list.'
    global coords
    coords.append([a,b,c])
    return

def coord_transform(sn, x, y, z):
    'Transforms the genetic unit spherical coordinates to the Lattice unit spherical coordinates.'
    Ld = {6:0, 9:1}
    if Ld[sn.Lattice.size]: #find the angles from the old lattice
        #This part is empty because I stopped needing to use lattice vectors.
        #If filled in, it would have to be complex matrix math that I don't want to do.
        pass 
    else: #find the angles from the new lattice, look in my notebook for diagrams
        a = np.pi/2
        a2, b2, c2 = np.radians(sn.Lattice[1])
        b0 = np.cos(a2)
        a0 = np.cos(b2)
        c0 = np.sqrt(a0**2+b0**2-2*a0*b0*np.cos(c2))
        A1 = a - (np.arcsin(b0*np.sin(c2)/c0))
        a1 = c0*np.sin(A1)
        b1 = c0*np.cos(A1)
        b3 = b1 + 1
        c1 = np.sqrt(a1**2+b3**2)
        c3 = np.sqrt(1-c0**2)
        CA = np.sqrt(c3**2 + c1**2)
        COA = np.arccos((CA**2-1-1)/-2)
        T = np.array([[np.cos(0), np.cos(a), np.cos(a)],\
                      [np.cos(a), np.cos(c2-a), np.cos(a)],\
                      [np.cos(b2), np.cos(COA), np.cos(a-np.arcsin(c3))]])
        new_coords = np.matmul(T, np.array([x,y,z]))
    return new_coords

def O_positions(sn, fn):
    'Determines O positions from structure_name and writes to structure fn.'
    polyseq = Polyseq(sn)
    z_pos_O = [1, 7, 9, 15, 17, 23]
    if sn.Lattice.size == 6: #makes a pymatgen lattice to get abc easily
        lat = mg.Lattice.from_lengths_and_angles(sn.Lattice[0],sn.Lattice[1])
    else:
        lat = mg.Lattice(sn.Lattice)
    for i in range(0, 36): #First write M+ oxygens
        laer = i // 6
        z_pos = z_pos_O[laer]
        z = z_pos_dict[z_pos]
        j = poly_dict[polyseq[laer]]
        k = j[i % 6]
        x = xy_pos_dict[k][0]
        y = xy_pos_dict[k][1]
        writeatom(x, y, z)
    for i in range(0,9): #write A- oxygens, carbonate first
        laer = i // 3 #specify the layer
        ox = i % 3 #specifies which oxygen to write
        ld = {0:sn.L1, 1:sn.L2, 2:sn.L3}
        z_pos = laer_z[laer]
        xy = xy_pos_dict[ld[laer].C.pos]
        td = {0:[0,2*np.pi/3,2*np.pi*2/3], 1:[2*np.pi/12, 2*np.pi*5/12, 2*np.pi*9/12]}
        pd = {0:np.pi/2, 1:np.radians(90-ld[laer].C.phi), 2:np.radians(90+ld[laer].C.phi)}
        x1 = np.sin(pd[ox])*np.cos(td[ld[laer].C.theta][ox]-(np.pi/6))
        y1 = np.sin(pd[ox])*np.sin(td[ld[laer].C.theta][ox])
        z1 = np.cos(pd[ox])
        x2, y2, z2 = coord_transform(sn, x1, y1, z1)
        x = xy[0] + x1*(1.28/lat.a)
        y = xy[1] + y1*(1.28/lat.b)
        z = z_pos_dict[z_pos] + (1.28/lat.c)*z2
        writeatom(x, y, z)
    for i in range(1,4): #write A- (water) oxygens
        ld = {1:sn.L1, 2:sn.L2, 3:sn.L3}
        wnum = ['W1', 'W2']
        if True in np.isnan(ld[i].W1): 
            wnum.remove('W1') 
        if True in np.isnan(ld[i].W2): 
            wnum.remove('W2')
        if len(wnum) == 0: #no water in layer, skip
            continue
        for j in wnum:
            w = {'W1': ld[i].W1, 'W2':ld[i].W2}
            z = z_pos_dict[laer_z[i-1]]
            try: xy = xy_pos_dict[w[j].pos]
            except: print(i, j, w[j])
            x = xy[0]
            y = xy[1]
            writeatom(x, y, z)
    return

def H_positions(sn, fn):
    'Determine H positions and write to structure.'
    polyseq = Polyseq(sn)
    z_pos_H = [2, 6, 10, 14, 18, 22]
    if sn.Lattice.size == 6: #makes a pymatgen lattice to get abc easily
        lat = mg.Lattice.from_lengths_and_angles(sn.Lattice[0],sn.Lattice[1])
    else:
        lat = mg.Lattice(sn.Lattice)
    a = np.radians(104.5/2)
    b = np.pi/6
    clocktuple = {0:(a, -a), 1:(a+b, -a+b), 2:(a+2*b, -a+2*b), 3:(a+3*b, -a+3*b)}
    global x; global y
    for i in range(0, 36): #First write M+ hydrogens
        laer = i // 6
        z_pos = z_pos_H[laer]
        z = z_pos_dict[z_pos]
        j = poly_dict[polyseq[laer]]
        k = j[i % 6]
        x = xy_pos_dict[k][0]
        y = xy_pos_dict[k][1]
        writeatom(x, y, z)
    for i in range(1,4): #write water hydrogens
        ld = {1:sn.L1, 2:sn.L2, 3:sn.L3}
        wnum = ['W1', 'W2']
        if True in np.isnan(ld[i].W1): wnum.remove('W1')
        if True in np.isnan(ld[i].W2): wnum.remove('W2')
        if len(wnum) == 0: #no water in layer, skip
            continue
        for j in wnum:
            w = {'W1': ld[i].W1, 'W2':ld[i].W2} #write first (and second) waters
            ct = clocktuple[w[j].theta]
            phi = np.radians(90-w[j].phi)
            for th in ct: #do for each water in ct
                xy = xy_pos_dict[w[j].pos]
                x1 = np.sin(phi)*np.cos(th-b)
                y1 = np.sin(phi)*np.sin(th)
                z1 = np.cos(phi)
                x2, y2, z2 = coord_transform(sn, x1, y1, z1)
                x = xy[0] + x2 * (0.9584 / lat.a)
                y = xy[1] + y2 * (0.9584 / lat.b)
                z = z_pos_dict[laer_z[i-1]] + (0.9584 / lat.c) * z2
                writeatom(x, y, z)
    return

def M_positions(sn, fn):
    'Determine Li and Al positions.'
    if sn.Polytope == 1:
        Li = [8, 17, 0, 9, 4, 13]
        Al = [1, 3, 10, 12, 5, 7, 14, 16, 2, 6, 11, 15]
    else:
        Li = [8, 17, 4, 13, 0, 9]
        Al = [1, 3, 10, 12, 2, 6, 11, 15, 5, 7, 14, 16]
    z_pos_M = [0, 8, 16]
    for i in range(0,6):
        laer = i // 2
        z = z_pos_dict[z_pos_M[laer]]
        x = xy_pos_dict[Li[i]][0]
        y = xy_pos_dict[Li[i]][1]
        writeatom(x, y, z)
    for i in range(0, 12):
        laer = i // 4
        z = z_pos_dict[z_pos_M[laer]]
        x = xy_pos_dict[Al[i]][0]
        y = xy_pos_dict[Al[i]][1]
        writeatom(x, y, z)
    return

def C_positions(sn, fn):
    'Determine C positions.'
    for i in range(1,4):
        ld = {1: sn.L1, 2:sn.L2, 3:sn.L3}
        z = z_pos_dict[laer_z[i-1]]
        xy = xy_pos_dict[ld[i].C.pos]
        x = xy[0]; y = xy[1]
        writeatom(x, y, z)
    return

def XRD_test(fn, s, genedex):
    '''Takes structure and makes XRD data, filters and analyzes, and adds the
    results to the genedex and then writes the Calc XRD file to the folder.'''
    a = xrds.get_xrd_pattern(s) #make xrd pattern
    c = np.stack((a.x, a.y), axis=-1) #combine into x,y coords
    c = c[c[:,1] > 10] #filter out peaks that are too small
    chklist=[]
    mpstring = 'x' #string of matching peaks
    for i in c[:,0]: #for each peak in the calc xrd:
        x = 0 #exptl peak counter
        for j in expt[:,0]: #compare to each exptl peak:
            if np.abs(i-j) <= expt[x,1]: #if peak is closer than err
                chklist.append(np.abs(i-j)) #note the diff from expt
                mpstring += str(x)
                break #move on to next calc peak
            else: #if peak is farther than err, check next exptl peak
                x += 1
                continue
    #After all calc peaks are checked:
    chkry = np.array(chklist)
    d = chkry.sum(0)/len(chkry)
    genedex.loc[fn, ('XRD', '', 'NoMP')] = len(set(mpstring))-1 #index Number of Matching Peaks
    genedex.loc[fn, ('XRD', '', 'MPerr')] = d #index Matching Peak error
    genedex.loc[fn, ('XRD', '', 'TotalPs')] = len(c) #index Total Peaks
    genedex.loc[fn, ('XRD', '', 'MPstring')] = mpstring
    return genedex

def Selection(fitdex, run, generation):
    'Sort and select the best and the randomly chosen extras for breeding.'
    sel = 0.35 #The best are chosen
    breeders = fitdex.nlargest(int(sel*len(fitdex)), 'fitsum')
    breeders.to_csv('./Genetics%s/Generation %s Breeders.csv' % (run, generation))
    a = len(breeders)*1.25 #add in some lucky extras for diversity
    while len(breeders) < a:
        breeders = pd.concat([breeders, fitdex.sample()])
    breedlist = list(breeders.index) #get list of fn's to pass into genedex
    for i in range(20): #duplicate best individual a bunch
        breedlist.append(breedlist[0])
    return breedlist

def Breed(breedlist): #Put Selection() in the args to pipe breedlist over
    'Mix up breeders and generate children. Assign the children to a genedex.'
    if float(run) >= 6.0: abc = True
    children = init_children(abc)
    kid = 0 #indexer
    for i in range(len(breedlist)): #p's chosen in order, m's at random
        p = dna_index[breedlist[i]]
        m = dna_index[breedlist[ra(len(breedlist))]]
        for j in range(np.random.choice([2,2,3,3])): #they have 2.25 kids
            children.loc[kid] = Child(p,m) #appends the child-list to the children-genedex
            kid += 1
    return children

def Child(p,m):
    'Given two parents, mix alleles to make the child. Whole anions are mixed, retaining their rotation.'
    inheritance = {0:p, 1:m} #coin-flipper
    poly = int(inheritance[ra(2)].Polytope); poly = Mutate(poly)
    child = [poly] #rna-esque list to hold nucleotides in order for the children-genedex
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
    s = np.random.random() #random seed
    if type(gene) == int: #mutate polytope to other value
        mut = [1,2]
        if s > 0.985:
            mut.remove(gene)
            gene = mut[0]
        else: pass
    elif type(gene) == np.ndarray: #mutate lattice paramaters with independent 1% chances
        if gene.size == 9: #test for old or new lattice
            mut = np.random.random() * 0.2 - 0.1 #keeps change small
            if s >= 0.99: #a uses x only
                gene[0][0] += mut
            if np.random.random() > 0.99: #b needs trig to shift coord axes in x and y
                r = np.sqrt((gene[1][0])**2 + (gene[1][1])**2)
                r += mut
                gene[1][0] += r*-0.5
                gene[1][1] += r*np.sqrt(3)/2
            if np.random.random() > 0.99: #c uses z only
                gene[2][2] += mut
            else: pass
        else:
            mut = np.random.random()*0.2-0.1
            for i in range(6):
                if np.random.random() >= 0.98:
                    gene[i//3][i%3] += mut
                else: pass
    else: #mutate molecular tuple
        mut = [1, -1, 3, -3]
        if s > 0.95:
            a,b,c = gene
            b += np.random.choice(mut)
            if b > 30: b = 30
            elif b < -30: b = -30 #keep phi within 30 deg.
            if s > 0.99:
                a += np.random.choice(mut)
                a = a % 18 #limit pos to valid numbers
            gene = M(a, b, c)
        else: pass
    return gene

def Ribosome(generation, run):
    'Read a csv index file and compile the genedex, given the run and generation.'
    if float(run) >= 6: abc = True
    else: abc = False
    genedex = init_genedex(abc)
    strindex = pd.read_csv('./Genetics%s/Genedex Gen %s.csv' % (run,generation), index_col=0)
    strindex = strindex[0:(len(strindex)-3)] #removes first two nan lines from bad MI conversion.
    strist = []
    strict = {}
    counter = 0
    for i in strindex:
        strist.append(strindex[i])
    for i in genedex:
        strict[counter] = i
        counter += 1
    for i in range(len(strindex.columns)):
        genedex[strict[i]] = strist[i]
    genedex = genedex.convert_objects(convert_numeric=True)
    return genedex

#Main program:
if generation == 0:
    genedex = Ribosome(generation, run) #1st run functionality
while generation <= 10: #set this number however many generations you want to run to.
    for fn in genedex.index:
        Transcription(fn, genedex) #Make DNA
        sn = dna_index[fn]
        coords = []
        O_positions(sn, fn) #Use DNA to write atoms to coords
        H_positions(sn, fn)
        M_positions(sn, fn)
        C_positions(sn, fn)
        XRD_test(fn, Structure_start(sn, fn, coords), genedex) #make and test structure
        fitdex = Fitness(fn, genedex) #Determine fitness
    children = Breed(Selection(fitdex, run, generation)) #Reproduction
    genedex.to_csv('./Genetics%s/Genedex Gen %s.csv' % (run,generation))
    genedex = children #Generation wrap-up and increment
    dna_index = {}
    fitdex = pd.DataFrame(columns=['matchfit', 'matchtot', 'fitsum'])
    generation += 1
