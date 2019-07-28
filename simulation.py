import math, csv
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import numpy as np
import random as rand
from texttable import Texttable


parameter = {};

parameter['varianceHD']     = 20
parameter['learnK']         = 0.01
parameter['learnKtrace']    = 0.01
parameter['traceN']         = 0.9#0
parameter['const0']         = 400
parameter['const1']         = 400*1.25
parameter['beta']           = 0.2
parameter['alpha']          = -2
parameter['wINH']           = 0.02

parameter['nrRotcells']     = 2
parameter['nrHDcells']      = 120

parameter['binary_thrsh_value'] = 0.3 # no longer used

#functions

def rate_sigmoid(val):
    return 1.0 / (1.0 + math.exp(-2*parameter['beta']*(val - parameter['alpha'])))
np_rate_sigmoid = np.vectorize(rate_sigmoid)

def rate_learning(val):
    return math.exp(-(val)**2/(2*(parameter['varianceHD'])**2))
np_rate_learning = np.vectorize(rate_learning)

#---------------------------
# HD, ROT cells and weights
#---------------------------

HDcells     = np.zeros(parameter['nrHDcells'])
ROTcells    = np.zeros(parameter['nrRotcells'])
HDweights   = np.zeros((parameter['nrHDcells'], parameter['nrHDcells']))
ROTweights  = np.zeros((parameter['nrRotcells'], parameter['nrHDcells'], parameter['nrHDcells']))

#converts cell index to prefered angle
def ctd(ind):
    return (360.0*ind)/(parameter['nrHDcells'])
def mrot(deg):
    return min(deg, 360.0-deg)

def training():
    global HDweights, ROTweights

    coolPhase   = 20
    nrRounds    = 10
    # rotate through all nrHDcells twice clock and anticlockwise
    for clock in [1,-1]:
        initDirection = 0
        # set rotation to clock or anticlockwise c^{ROT}
        if(clock == -1):
            ROTcells[0] = 1
            ROTcells[1] = 0
        else:
            ROTcells[0] = 0
            ROTcells[1] = 1

        #iterate over the prefered directions
        prevHDrates = np.zeros(parameter['nrHDcells'])
        coolPhaseactive = True

        #first have a cooling phase in which nothing actually gets updated - then have the real training phase
        for cellInd in range(nrRounds*parameter['nrHDcells']+coolPhase):
            direction = (initDirection + clock*cellInd) %parameter['nrHDcells']
            # how far away from pref. direction
            s               = np.array([mrot(abs(ctd(i) - ctd(direction))) for i in range(parameter['nrHDcells'])])
            HDrates         = np.array([(rate_learning(s[i])) for i in range(parameter['nrHDcells'])])
            traceHDrates    = np.array([(1 - parameter['traceN'])*HDrates[i] + parameter['traceN']*prevHDrates[i] for i in range(parameter['nrHDcells'])])
            prevHDrates     = np.copy(traceHDrates)

            if(cellInd > coolPhase):
                coolPhaseactive = False
            if coolPhaseactive:
                continue

            # HD cell connections
            for i in range(parameter['nrHDcells']):
                #HDweights [i][j] += learnK * traceHDrates[i]*traceHDrates[j]
                HDweights[i] = HDweights[i] + parameter['learnK']*traceHDrates[i]*traceHDrates
                #HDweights[i] = HDweights[i] + learnK*HDrates[i]*HDrates


            #train rotation connections as well
            for k in range(parameter['nrRotcells']):
                for i in range(parameter['nrHDcells']):
                    ROTweights[k][i] = ROTweights[k][i] + parameter['learnKtrace']*HDrates[i]*ROTcells[k]*traceHDrates
                    #ROTweights[k][i][j] = learnKtrace * HDrates[i] * traceHDrates[j] * ROTcells[k]


    #RENORMALIZE weights
    sums = np.linalg.norm(HDweights,axis=1)
    HDweights = HDweights / sums[:np.newaxis]

    sums = np.linalg.norm(ROTweights, axis=2)
    ROTweights = ROTweights / sums[:,:,np.newaxis]


def rate_formula_HD(i,I):
    return parameter['const0']/parameter['nrHDcells'] * np.sum(np.multiply(HDweights[i] - parameter['wINH'], np_rate_sigmoid(HDcells))) + I
    #sum([(HDweights[i][j] - wINH) * rate_sigmoid(HDcells[j]) for j in range(parameter['nrHDcells'])]) + I[i]
def rate_formula_ROT(i):
    return parameter['const1']/(parameter['nrHDcells'] * parameter['nrRotcells']) * np.sum([np.sum((ROTcells[k]) * np.multiply(ROTweights[k][i], np_rate_sigmoid(HDcells))) for k in range(parameter['nrRotcells'])])
#    sum( [(ROTweights[k][i][j] * rate_sigmoid(HDcells[j]) * rate_sigmoid(ROTcells[k])) for j in range(parameter['nrHDcells']) for k in range(nrRotcells)])
def rate_formula(i, I):
    return -HDcells[i] + rate_formula_HD(i,I) + rate_formula_ROT(i)

def binary_thrsh(val):
    return val
    if val < parameter['binary_thrsh_value']:
        return 0.0
    else:
        return 1.0
def max_zero(val):
    return max(val,0)
def newZero(arr):
    return [max(i,0) for i in arr]
np_max_zero = np.vectorize(max_zero)
np_binary_thrsh = np.vectorize(binary_thrsh)

def rate_thrsh_part(k):
    return newZero(np_binary_thrsh(np_rate_sigmoid(HDcells)) - ROTcells[k] )

def rate_formula_ROT_2(i):
    return parameter['const1']/(parameter['nrHDcells'] * parameter['nrRotcells']) * np.sum([np.sum(np.multiply(ROTweights[k][i], rate_thrsh_part(k) )) for k in range(parameter['nrRotcells'])])

def rate_formula_2(i,I):
    return -HDcells[i] + rate_formula_HD(i,I) + rate_formula_ROT_2(i)

np_rate_formula     = np.vectorize(rate_formula)
np_rate_formula_2   = np.vectorize(rate_formula_2)



plotter = []
lines   = []
fig = []
def init_plot():
    global plotter, lines, fig
    plotter = []
    lines = []
    fig = []
def _plot(data, labels, iter):
    global plotter, lines, fig
    # init
    n = len(data)
    if plotter == []:
        plt.ion()
        fig = plt.figure(figsize=(10,10))
        xx = max(len(data)//3,1)
        yy = max(len(data)//xx,1)
        for i in range(n):
            subi = int(str(xx)+str(yy)+str(i+1))
            ax = fig.add_subplot(xx,yy,i+1)
            ax.title.set_text(labels[i])
            plotter.append(ax)
            line1, = ax.plot([i for i in range(len(data[i]))],data[i],'-o',alpha=0.8)
            lines.append(line1)
        plt.show()

    fig.suptitle('Iteration: {}'.format(iter))
    for i in range(n):
        lines[i].set_ydata(data[i])
        if np.min(data[i])<=lines[i].axes.get_ylim()[0] or np.max(data[i])>=lines[i].axes.get_ylim()[1]:
            plotter[i].set_ylim(bottom=np.min(data[i])-np.std(data[i]))
            plotter[i].set_ylim(top=np.max(data[i])+np.std(data[i]))

    plt.pause(0.1)

def simulateStringer(totTime):
    global parameter

    timestep = 0.1
    time = 0
    #time
    line1 = []
    line2 = []
    ROTcells[0] = 0
    ROTcells[1] = 0
    I0 =    np.zeros(parameter['nrHDcells'])
    HDcells = np.zeros(parameter['nrHDcells'])
    parameter['wINH'] = 0.3 * np.max(HDweights)
    s             = np.array([mrot(abs(ctd(i) - ctd(20))) for i in range(parameter['nrHDcells'])])
    Iorig         = 10*np.array([((rate_learning(s[i]))) for i in range(parameter['nrHDcells'])])
    enumArray   = np.array([i for i in range(parameter['nrHDcells'])])


    while time < totTime:

        #print(time)
        #HDcells are h_i



        I = []
        if time < 2 and time > 0:
            I = Iorig
            #I = I0
        else:
            I = I0
        if time > 2 and time < 10:
            ROTcells[0] = 0
            ROTcells[1] = 1
        elif time > 10:
            ROTcells[0] = 0.5
            ROTcells[1] = 0


        diffActiv   = np_rate_formula(enumArray, I)
        #[rate_formula(i,I0) for i in range(parameter['nrHDcells'])]
        HDcells     = HDcells + timestep*diffActiv
        #[HDcells[i] + diffActiv[i]*timestep for i in range(parameter['nrHDcells'])]
        if time > 0 and (int(time)%2 == 0):
            print(time)
            #line2        = live_plotter([i for i in range(parameter['nrHDcells'])], [rate_sigmoid(i) for i in HDcells], line2, str(int(time*100)+1))
#            line1, line2        = live_plotter([i for i in range(parameter['nrHDcells'])],[rate_formula_ROT(i) for i in range(parameter['nrHDcells'])], [(i) for i in diffActiv], line1,line2, str(int(time*100)+1))
#            line1, line2        = live_plotter([i for i in range(parameter['nrHDcells'])],[rate_sigmoid(i) for i in HDcells], [(i) for i in diffActiv], line1,line2, str(int(time*100)+1))
            _plot([
                    [rate_sigmoid(i) for i in HDcells],
                    [i for i in diffActiv],
                    [rate_formula_ROT(i) for i in range(parameter['nrHDcells'])],
                    [rate_formula_HD(i,I[i]) for i in range(parameter['nrHDcells'])],
                    [i for i in ROTcells],
                    [i for i in HDcells],
            ], ['HDcells rate', 'diffActiv', 'diff ROT', 'diff HD', 'rate ROT', 'HDcells'], str(int(time*100)+1))
        time        = time + timestep

def trackCenter():
    return np.where(HDcells == np.amax(HDcells))[0][0]

def trackCenterArray():
    a = np.zeros(parameter['nrHDcells'])
    a[trackCenter()] = 1
    return a

def selectRotation(time, rotationInp):
    for begin, end, rot0, rot1 in rotationInp:
        if begin <= time and time < end:
            return rot0, rot1
    return rotationInp[-1][2],rotationInp[-1][3]

def exportTrackingData(filename, rotationInp, centerHistory,timestep):
    # export a table with [time, rot0, rot1, center]

    #create Table
    t = [[i, selectRotation(i*timestep, rotationInp)[0], selectRotation(i*timestep,rotationInp)[1], centerHistory[i]] for i in range(len(centerHistory))]

    with open(filename, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for r in t:
            writer.writerow(r)

def parameterOverview(loc):
    print("======================================")
    print("|         Simulation Start           |")
    print("======================================")


    t = Texttable()
    t.add_rows([x for x in [['Parameter', 'Value']]+[[p, parameter[p]] for p in parameter] ])
    print(t.draw())
    t = Texttable()
    t.add_rows([x for x in [['Parameter', 'Value']]+[[p, loc[p]] for p in loc] ])
    print(t.draw())
    t = Texttable()
    t.add_rows([x for x in [['Start', 'End', 'rot0', 'rot1']]+[[a,b,c,d] for a,b,c,d in loc['rotationInp']] ])
    print(t.draw())
    t = Texttable()
    t.add_rows([x for x in [['Start', 'End', 'visual Center']]+[[a,b,c] for a,b,c in loc['visualInp']] ])
    print(t.draw())

def simulate(visualInp = [[]] ,rotationInp = [[0,10,1,1]],simulationTime = 10, timestep = 0.1, verbose = False,progress = True,filename = 'simulation.csv'):
    loc = locals()
    global HDcells, ROTcells, parameter

    # set simulation variables
    #==================================================================
    time = 0
    HDcells     = np.zeros(parameter['nrHDcells'])-10
    parameter['wINH']        = 0.5 * np.max(HDweights)
    enumArray   = np.array([i for i in range(parameter['nrHDcells'])])
    #==================================================================

    # Tracking Data
    #==================================================================
    centerHistory   = [0 for i in range(100)]
    #==================================================================


    if verbose:
        parameterOverview(loc)

    progressPer = 0
    #Simulation Loop
    #==================================================================
    while time < simulationTime:

        newProgress = int(time/simulationTime*100)
        if newProgress > progressPer:
            print(newProgress)
            progressPer = newProgress
        # set external Inp
        #==================================================================
        I = np.zeros(parameter['nrHDcells'])
        for begin,end, visualCenter in visualInp:
            if begin <= time and time < end:
                I = 10* np.array([rate_learning( mrot(abs(ctd(i) - ctd(visualCenter))) ) for i in range(parameter['nrHDcells'])])

        ROTcells[0], ROTcells[1] = selectRotation(time, rotationInp)

        #==================================================================

        # calculate diff. Step for simulation
        #==================================================================
        diffActiv   = np_rate_formula_2(enumArray, I)
        #==================================================================

        # VERBOSE
        #==================================================================
        if time > 0 and (int(time)%2 == 0):
            if(verbose):
                _plot([
                        [rate_sigmoid(i) for i in HDcells],
                        [i for i in diffActiv],
                        [rate_formula_ROT_2(i) for i in range(parameter['nrHDcells'])],
                        [rate_formula_HD(i,I[i]) for i in range(parameter['nrHDcells'])],
                        [i for i in ROTcells],
                        [i for i in HDcells],
                        [i for i in I],
                        [i for i in newZero(np_binary_thrsh(np_rate_sigmoid(HDcells)) - ROTcells[0] )],
                        [i for i in rate_thrsh_part(1)],
                        [i for i in (np_binary_thrsh(np_rate_sigmoid(HDcells)) - ROTcells[1]) ],#trackCenterArray(),
                        centerHistory[-101:-1],
                        [i for i in (np_binary_thrsh(np_rate_sigmoid(HDcells)) - ROTcells[0]) ],
                ], ['HDcells rate', 'diffActiv', 'diff ROT', 'diff HD', 'rate ROT', 'HDcells', 'ext. visual Inp', 'ROT thrsh 0', 'rate thrsh 1', 'center', 'c', 'c'],
                    str(int(time*100)+1))
        #==================================================================


        #update simulation variables
        #==================================================================
        centerHistory.append(trackCenter())

        HDcells     = HDcells + timestep*diffActiv
        time        = time + timestep
        #==================================================================

    # Prepare Data for analysis
    #==================================================================
    exportTrackingData(filename,rotationInp, centerHistory[100:],timestep)
    #==================================================================
def plotWeights(index):
    plt.plot([HDweights[i][index] for i in range(parameter['nrHDcells'])])
    plt.ylabel("$w_{" + str(index)+ ",j}^{HD}$" )
    plt.xlabel("$j$")
    plt.show()

    plt.ylabel("$w_{i," + str(index)+ ",k}^{ROT}$" )
    plt.xlabel("$i$")

    a0 = [ROTweights[0][i][index] for i in range(parameter['nrHDcells'])]
    a1 = [ROTweights[1][i][index] for i in range(parameter['nrHDcells'])]
    print(sum([HDweights[index][i]**2 for i in range(parameter['nrHDcells'])]))
    print(sum([HDweights[i][index]**2 for i in range(parameter['nrHDcells'])]))
    print(sum([ROTweights[0][index][i] ** 2 for i in range(parameter['nrHDcells'])]))
    print(sum([ROTweights[1][index][i] ** 2 for i in range(parameter['nrHDcells'])]))
    print(sum([ROTweights[0][i][index] ** 2 for i in range(parameter['nrHDcells'])]))
    print(sum([ROTweights[1][i][index] ** 2 for i in range(parameter['nrHDcells'])]))

    line1, = plt.plot(a0, label="$w_{i," + str(index)+ ",0}^{ROT}$")
    plt.plot(a1, label="$w_{i," + str(index)+ ",1}^{ROT}$")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})

    plt.show()
def showWeights(index,diff,rep):
    for _ in range(rep):
        plt.plot(HDweights[index])
        index = (index + diff)%parameter['nrHDcells']
    plt.ylabel('weights of HD cell ' + str(index))
    plt.show()

def showROTWeights(rot, index,diff,rep):
    for _ in range(rep):
        plt.plot(ROTweights[rot][index])
        index = (index + diff)%parameter['nrHDcells']
    plt.ylabel('weights of ROT cell ' + str(rot)+ ',' + str(index))
    plt.show()

def showROTweights2(index, diff, rep):
    for _ in range(rep):
        index = index % parameter['nrHDcells']
        plt.plot(ROTweights[0][index])
        plt.plot(ROTweights[1][index])
        index = (index + diff)%parameter['nrHDcells']
    plt.ylabel('weights of ROT cell ' + str(index))
    plt.show()

# generate n random orders with normal distributed time duration N(timexp, timevar)
def experiment(n, timeexp, timevar):
    # choose random direction and random speed
    # randomly flip for direction, then do gaussian for the speed
    np.random.seed(2)
    s = np.random.normal(0.5, 0.2, n)
    s = [round(x,2) for x in s if x >= 0 and x <= 1]
    s = [(x, 1.0) if np.random.randint(0,2) == 0 else (1.0, x) for x in s ]
    l = len(s)
    times = np.random.normal(timeexp,timevar,l)
    times = [round(x,0) for x in times if x > 0]
    l = min(len(s), len(times))
    times   = times[:l]
    s       = s[:l]

    prep = [[0,10,1,1]]

    for x in range(l):
        pre = prep[-1][1]
        time = times[x]
        rot0, rot1 = s[x]
        prep.append([pre, pre+time, rot0, rot1])

    return prep

# generate 2n random orders evenly distr. along [0,1] for rot. inp, specify time duration
def experiment_const_speed(n, t, pause_t):
    prep = [[0,10,1,1]]
    speeds = [round(i/n,2) for i in range(n)]
    for i in speeds:
        d = t
        dd = pause_t
        # right
        pre = prep[-1][1]
        prep.append([pre, pre+d, 1,i])
        pre = prep[-1][1]
        prep.append([pre, pre+dd, 1, 1])
        #left
        pre = prep[-1][1]
        prep.append([pre, pre+d, i, 1])
        pre = prep[-1][1]
        prep.append([pre, pre+dd, 1, 1])
    return prep




#==========================================
# setup the weights with the learning rules
#==========================================

training()


#showWeights(0, 10, 10)
#showWeights(0, 2, 10)
#showWeights(80, 2, 10)
#showWeights(0, 1, 100)
#showROTWeights(0,0,20,5)
#showROTWeights(1,0,20,5)
#showROTweights2(0,40,3)
#showROTweights2(10,40,3)
#showROTweights2(20,40,3)

#plotWeights(60)


#=============================================
# perform the simulation
#=============================================
# don't forget to define the filename for the export of the data

exportFileName = 'export.csv'
# prep
orders = experiment_const_speed(20,2,0)
orders = experiment(10,5,1)

visualInp = [[0,2,0]] # set the visual input

simulate(
        visualInp       = visualInp,
        rotationInp     = orders,
        simulationTime  = orders[-1][1],
        timestep        = 0.1,
        verbose         = False,
        progress        = True,
        filename        = exportFileName
        )
