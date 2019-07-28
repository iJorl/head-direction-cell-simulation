import csv, math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def getFunction(stats):
    #fix a distance and then plot the speed relation

    rotations = sorted([r for r in stats])
    intervals = [i for i in stats[rotations[0]]]

    n = len(intervals)
    fig = plt.figure(figsize=(10,10))

    xx = 1
    yy = n

    intervals = [100,100]
    print("in function 2")
    for i in range(2):
        ax = fig.add_subplot(1,2,i+1)
        ax.title.set_text("$\Delta t =" +str( intervals[i])+ "$")
        #ax.set_ylabel("speed")
        #ax.set_xlabel("rotation input")
        x = [min(i[0],i[1]) for i in rotations]
        print(i)
        #generate y data
        y = []
        e = []

        raw_data_x = []
        raw_data_y = []

        rot0 = [r for r in rotations if r[0] <= r[1]]
        rot1 = [r for r in rotations if r[0] >= r[1]]
        #split them in left and right rotation
        for rot in rot0:
            a = np.array(stats[rot][intervals[i]])/intervals[i]*10
            y.append(np.mean(a))
            e.append(np.std(a))
            if(np.isnan(np.mean(a)) or min(rot[0], rot[1]) > 0.9):
                continue
            raw_data_x.append(min(rot[0], rot[1]))
            raw_data_y.append(abs(np.mean(a)))
        ax.errorbar([min(i[0],i[1]) for i in rot0], y, e, label="$r_0^{ROT}$")
        y = []
        e = []
        x = [min(i[0],i[1]) for i in rot1]
        for rot in rot1:
            a = np.array(stats[rot][intervals[i]])/intervals[i]*10
            y.append(np.mean(a))
            e.append(np.std(a))
            if(np.isnan(np.mean(a)) or min(rot[0], rot[1]) > 0.9):
                continue
            raw_data_x.append(min(rot[0], rot[1]))
            raw_data_y.append(abs(np.mean(a)))

        ax.errorbar(x, y, e, label="$r_1^{ROT}$")
        xn = [i/50 for i in range(50)]

        if i == 0:
            ax.plot(xn, [10*linF(xx) for xx in xn ], color='red')
            ax.plot(xn, [-10*linF(xx) for xx in xn ], color='red', label='linear $\Psi$')
        else:
            ax.plot(xn, [10*quadF(xx) for xx in xn ], color='green')
            ax.plot(xn, [-10*quadF(xx) for xx in xn ], color='green', label='quadratic $\Psi$')


        if len(raw_data_x) == 0 or len(raw_data_y) == 0:
            continue

        #print(raw_data_x)
        #print(raw_data_y)
        train_y = np.array(raw_data_y).reshape(-1,1)
        train_x = np.array(raw_data_x).reshape(-1,1)

        reg = LinearRegression().fit(train_x, train_y)
        reg.coef_ = reg.coef_.astype(dtype=np.float64)
        print(reg.coef_)

        #ax.plot(x, [reg.coef_[0]*(1-xx) for xx in x ])



        polynomial_features= PolynomialFeatures(degree=2)
        x_poly = polynomial_features.fit_transform(train_x)
        reg = LinearRegression().fit(x_poly, train_y)
        print(reg.coef_)
        newx = polynomial_features.fit_transform(np.array(x).reshape(-1,1))
        predy = reg.predict(newx)
        #ax.plot(x, predy)

        #print(newx)
        #print(predy)
        #print(reg.get_params())

        c0 = reg.coef_[0]
        x0 = newx[0]
        y0 = predy[0]
        m = y0 - sum([x0[i]*c0[i] for i in range(len(c0))])
        print("c0 should be", m)


        ax.legend()

    fig.text(0.5, 0.06, 'rotation input', ha='center')
    fig.text(0.06, 0.5, 'speed', va='center', rotation='vertical')

    plt.show()

minDist = 3

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

def plot_speed_distance(stats):
    #fix a distance and then plot the speed relation

    rotations = sorted([r for r in stats])
    intervals = [i for i in stats[rotations[0]]]

    n = len(intervals)
    fig = plt.figure(figsize=(10,10))

    xx = 1
    yy = n


    for i in range(4):
        ax = fig.add_subplot(2,2,i+1)
        ax.title.set_text("$\Delta t =" +str( intervals[i])+ "$")
        #ax.set_ylabel("speed")
        #ax.set_xlabel("rotation input")
        x = [min(i[0],i[1]) for i in rotations]

        #generate y data
        y = []
        e = []

        rot0 = [r for r in rotations if r[0] <= r[1]]
        rot1 = [r for r in rotations if r[0] >= r[1]]
        #split them in left and right rotation
        for rot in rot0:
            a = np.array(stats[rot][intervals[i]])/intervals[i]*10
            y.append(np.mean(a))
            e.append(np.std(a))
        ax.errorbar([min(i[0],i[1]) for i in rot0], y, e, label="$r_0^{ROT}$")
        y = []
        e = []
        x = [min(i[0],i[1]) for i in rot1]
        for rot in rot1:
            a = np.array(stats[rot][intervals[i]])/intervals[i]*10
            y.append(np.mean(a))
            e.append(np.std(a))
        ax.errorbar(x, y, e, label="$r_1^{ROT}$")
        ax.legend()

    #plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    #plt.grid(False)

    #plt.ylabel("speed")
    #plt.xlabel("rotation input")
    fig.text(0.5, 0.06, 'rotation input', ha='center')
    fig.text(0.06, 0.5, 'speed', va='center', rotation='vertical')
    plt.show()

def plot_speed_over_time(stats):
    # for a certain speed and a certain distance,
    # check if stays the same over specified intervals

    #evenly divide into x parts
    fig = plt.figure(figsize=(10,10))
    xx = 1
    yy = 1

    ax = fig.add_subplot(xx,yy,1)

    # given rotation r, and measuring interval d, and measure for k divided segments
    r = (0, 1)
    d = 100
    l = len(stats[r][d])
    k = int(l/20)
    ax.title.set_text(str(r) + " " + str(d) + " " + str(k))


    x = [i*l/k for i in range(k)]
    y = []
    e = []
    #generate y data
    for kk in range(k):
        a = np.array(stats[r][d][int(kk*l/k):int((kk+1)*l/k)])/d*10

        y.append(np.mean(a))
        #e.append(np.std(a))
        e.append(0)
    ax.errorbar(x, y, e)

    plt.show()

def plot_interval_speed(stats):

    n = len(stats)
    fig = plt.figure(figsize=(10,10))
    xx = 1
    yy = n

    rotations = [r for r in stats]

    for i in range(n):
        ax = fig.add_subplot(xx,yy,i+1)
        ax.title.set_text(str(rotations[i]))

        x = [vals for vals in stats[rotations[i]]]
        y = []
        e = []
        for interval in stats[rotations[i]]:
            a = np.array(stats[rotations[i]][interval])/interval*10
            y.append(np.mean(a))
            e.append(np.std(a))

        ax.errorbar(x, y, e)


    plt.show()

def getData(filename):
    data = []
    with open(filename) as file:
        csv_reader = csv.reader(file, delimiter=',')
        line_count = 0
        data = [row for row in csv_reader if row != []]
    return [[float(a),float(b),float(c),float(d)] for a,b,c,d in data[1:]]

def dist(a,b):
    if(abs(b-a) > minDist):
        if b-a < 0:
            return (b-a)+120
        else:
            return (b-a)-120
    return b-a

def linF(x):
    return (1-x)*3.5/10

def quadF(x):
    return max((3.17+ (x**1)*-1.094 + ((x)**2)*-2.65)/10,0)

def rot_speed_lin(a, b):
    if a == 1:
        return linF(b)
    else:
        return -linF(a)
def rot_speed_quad(a, b):
    if a == 1:
        return quadF(b)
    else:
        return -quadF(a)
def plot_predict_center_over_time(data):
    origin = 0
    predicted = []
    x = []
    y = []
    for a,b,c,d in data:
        origin += rot_speed(b,c)
        if origin > 120:
            origin = origin - 120
        if origin < 0:
            origin = origin + 120
        predicted.append(origin)
        x.append(a)
        y.append(d)

    fig = plt.figure(figsize=(10,10))
    xx = 1
    yy = 1
    n = 1

    for i in range(n):
        ax = fig.add_subplot(xx,yy,i+1)
        ax.title.set_text("predicted")


        ax.plot(x, y)
        ax.plot(x, predicted)

    plt.show()

def forceInterval(x):
    if x > 120:
        return x - 120
    if x < 0:
        return x + 120
    return x

def error_over_time(data):

    #data = data[:5000]

    predicted_lin   = [0]
    predicted_quad  = [0]

    x = [0]
    y = [0]
    error_lin       = [0]
    error_quad      = [0]

    errorSum_lin    = [0]
    errorSum_quad   = [0]
    for a,b,c,d in data:

        pred_lin    = forceInterval( predicted_lin[-1] + rot_speed_lin(b,c))
        pred_quad   = forceInterval( predicted_quad[-1]+ rot_speed_quad(b,c))



        dO      = dist(y[-1], d)
        dP_lin  = dist(predicted_lin[-1], pred_lin)
        dP_quad = dist(predicted_quad[-1], pred_quad)

        error_lin.append((dO-dP_lin))
        error_quad.append((dO-dP_quad))

        errorSum_lin.append(errorSum_lin[-1]+error_lin[-1])
        errorSum_quad.append(errorSum_quad[-1]+error_quad[-1])


        predicted_lin.append(pred_lin)
        predicted_quad.append(pred_quad)

        x.append(a)
        y.append(d)

    fig = plt.figure(figsize=(10,10))
    xx = 2
    yy = 1
    n = 2
    for i in range(n):
        ax = fig.add_subplot(xx,yy,i+1)
        #ax.title.set_text("predicted")

        #ax.plot(x, y, color='blue', label='measured')
        if i % 2 == 0:
            #ax.plot(x, predicted_lin, color='red', label='predicted')
            ax.plot(x, errorSum_lin, color='red', label='error linear $\Psi$')
            ax.plot(x, errorSum_quad, color='green', label='error quadratic $\Psi$')
            ax.set_title("linear $\Psi$")
        else:
            #ax.plot(x, predicted_quad, color='green', label='predicted')
            ax.plot(x, errorSum_quad, color='green', label='error quadratic $\Psi$')
            ax.set_title("quadratic $\Psi$")
        ax.legend(loc='lower left')
    fig.text(0.5, 0.06, 'time', ha='center')
    #fig.text(0.06, 0.5, 'head direction', va='center', rotation='vertical')
    fig.text(0.06, 0.5, 'accumulated error', va='center', rotation='vertical')
    plt.show()

def analyse(filename):
    data = getData(filename)

    # calculate distance between points, can then take time intervals

    distance = [dist(data[i][3], data[i+1][3]) for i in range(len(data)-1)]
    # find all the rotational inputs

    rotationInp = list(set([(b,c) for a,b,c,d in data]))

    intervals = [5, 10,50,100,150]
    #intervals = [10,50,100,200,500]
    stats = {}
    for rot in rotationInp:
        stats[rot] = {}
        for interv in intervals:
            stats[rot][interv] = []

    for interval in intervals:
        for i in range(len(data)-interval):
            # check if have same rotationInp
            rInp = list(set([(b,c) for a,b,c,d in data[i:(i+interval)]]))
            if(len(rInp) != 1):
                continue
            coveredDist = sum(distance[i:(i+interval)])

            stats[rInp[0]][interval].append(coveredDist)

    print(max(distance), min(distance))
    #plot_interval_speed(stats)
    #plot_predict_center_over_time(data)
    plot_speed_distance(stats)
#    plot_speed_over_time(stats)
    getFunction(stats)
    error_over_time(data)

#====================================
# load the file exported for analysis
#====================================

analyse('sample_export.csv')

#analyse('eco_bothspeeds_20_20_0_c2_625.csv')
#analyse('eco_bothspeeds_20_20_0_c2_400.csv')

#analyse('eco_bothspeeds_20_20_0.csv')
#analyse('eco_bothspeeds_20_20_5.csv')
#analyse('eco_bothspeeds_20_200_0.csv')
