import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import os,sys,time
import math
import tensorflow as tf


def rescale(x, a, b, c, d):
    """
    Rescales variable from [a, b] to [c, d]
    """
    return c + ((d - c) / (b - a)) * (x - a)


def load_raw_data(base_name, num_clones):
    """

    :param base_name -  :
    :return:
    """
    data = []
    for i in range(num_clones):
        current_file_name = base_name + str(i) + ".txt"
        f = open(current_file_name, "r")
        print (current_file_name)
        cont = []
        if f.mode == 'r':
            with open(current_file_name) as input_file:
                for line in input_file:
                    line = line.strip()
                    for number in line.split():
                        cont.append(float(str.strip(str.strip(number, ']'), '[')))

            data.append(cont)
    np.asarray(data, dtype=float)
    return data

def schwefel(x, xmin=-1, xmax=1):
        """
        https://www.sfu.ca/~ssurjano/schwef.html
        """
        x = rescale(x, xmin, xmax, -500, 500)
        d = tf.to_float(tf.shape(x)[1])
        x = tf.sin(tf.abs(x) ** 0.5) * x
        result = 418.9829 * d - tf.reduce_sum(x, axis=1)

def load_data(base_name):
    """

    :param base_name -  :
    :return:
    """
    if base_name[-4]=='e':

        lr = float(base_name[-5:])
        if lr == 5e-5:
            divider = int(16e-4 / lr)

    else:
        lr = float(base_name[-6:])
    #if base_name[0] =='1':


    divider = int(16e-4/lr)

    divider=1
    data = []
    result_data=[]

    for i in range(1):
        current_file_name = base_name + ".txt"
        f = open(current_file_name, "r")
        print (current_file_name)
        cont = []
        if f.mode == 'r':
            with open(current_file_name) as input_file:
                for line in input_file:
                    line = line.strip()
                    for number in line.split():

                        cont.append(float(str.strip(str.strip(str.strip(number, ']'), '['),',')))

            data.append(cont)
    counter=0
    """
    d=[]
    data_=[]
    for example in data :
        counter+=1
        data_.append(example)
        if counter!=0 and counter%49==0:
            d.append(data_)
            data_=[]
    data=d
    """


    #np.asarray(data, dtype=float)
    """
    #data  = np.mean(data,axis=0)
    #ty = math.exp(0.1)-1
    #print ty
    #data = data - ty
    k = 0
    for j in data:
        if k % divider == 0:
            k += 1
            result_data.append(j)
        else:
            k += 1
    """

    return data


def log_mean(container):
    """

    :param container:
    :return:
    """
    log = np.log(container)
    l_mean = np.mean(log,axis=0)
    result = np.exp(l_mean)
    return result

def plot(names,num_clones,num_iterations):
    out_path = "./results"
    X = np.arange(0, num_iterations)
    fig = plt.figure()
    matplotlib.rcParams.update({'font.size': 18})
    fig.set_size_inches(12.8, 12.8)
    ax = fig.gca(yscale="log")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    for name in names:
        raw_data = load_data(name,num_clones)
        data = log_mean(raw_data)
        plt.plot(X, data)
    ax.legend() #now it made by hands need to be automatical
    plt.savefig(os.path.join(out_path, "1.png"))#now it made by hands need to be automatical
    plt.close()



def plot_data(names,num_iterations):
    out_path = "./results"
    X = np.arange(0, num_iterations)

    fig = plt.figure()
    matplotlib.rcParams.update({'font.size': 18})
    fig.set_size_inches(12.8, 12.8)
    ax = fig.gca(yscale="log")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    colors = []
    # colors_fb = []
    #color1 = [1, 0, 0, 1.2]
    #color2 = [0, 0, 1, 1.2]
    """
    for i in range(5):
        color1[3]=color1[3]-0.2
        print color1
        color2[3]=color2[3]-0.2
        colors.append(color1)
        colors.append(color2)
    """
    #colors=['#ffff14','#00035b','#fffe7a','#0343df','#ffffc2','#75bbfd','#dbb40c','#c6fcff','#fac205','#d0fefe']
    #ax.set_color_cycle(colors)
    for name in names:
        data = load_data(name)
        data=np.mean(data,axis=0)
        #data = data[np.logical_not(np.isnan(data))]
        #data.dropna(inplace=True)
        #data = savgol_filter(data, 401, 1)
        #coord_data = np.genfromtxt(name + '.csv', delimiter=',')





        #data_ = savgol_filter(data, 101, 4)
        plt.plot(X, data)
    #ax.legend(["7.5e-5","16e-4 fb","8e-4 ","8e-4 fb","4e-4 ","4e-4 fb","2e-4 ","2e-4 fb","1e-4 ","1e-4 fb" ])  # now it made by hands need to be automatical
    ax.legend(["16","8","4","2","1","1/2","1/4","1/8","1/16"])
    plt.savefig(os.path.join(out_path, "momentum_mean_.png"))  # now it made by hands need to be automatical
    plt.close()

def plot_coords_data(names,num_iterations):
    out_path = "./results/Plots/embed1"
    X = np.arange(0, num_iterations)
    min_loss = load_data("16.0*r_momentum1_min_loss_[50, 50, 1]schwefel0.00375")
    mean_loss = load_data("16.0*r_momentum1_mean_loss_[50, 50, 1]schwefel0.00375")

    sess = tf.Session()
    # data = savgol_filter(data, 401, 1)
    _x = np.linspace(-1, 1, 1000, dtype=np.float32)
    x = -500 + (500) * (_x + 1)
    x = np.stack([x], 1)
    _y = 418.9829 * tf.to_float(tf.shape(x)[0]) - tf.reduce_sum(tf.sin(tf.abs(x) ** 0.5) * x, axis=1)
    y = sess.run(_y)
    sess.close()
    tf.reset_default_graph()

    colors = []
    # colors_fb = []
    #color1 = [1, 0, 0, 1.2]
    #color2 = [0, 0, 1, 1.2]
    """
    for i in range(5):
        color1[3]=color1[3]-0.2
        print color1
        color2[3]=color2[3]-0.2
        colors.append(color1)
        colors.append(color2)
    """
    #colors=['#ffff14','#00035b','#fffe7a','#0343df','#ffffc2','#75bbfd','#dbb40c','#c6fcff','#fac205','#d0fefe']
    #ax.set_color_cycle(colors)

    coord_data = []
    np.asarray(coord_data)
    cr=0
    for name in names:
        coord_data_ = np.genfromtxt(name + '.csv', delimiter=',')
        if cr==0:
            coord_data= np.genfromtxt(name + '.csv', delimiter=',')
            print cr
            cr = 1
        else:
            print cr
            coord_data = np.concatenate((coord_data,coord_data_),axis=1)
        #data = load_data(name)
    if name[0] == 'd':
        flag = "deep"
    else:
        flag = 'n'
    # coord_data = np.genfromtxt(name + '.csv', delimiter=',')
    #print coord_data[31999]

    X1 = coord_data[0]

    coords = X1
    plt.plot(y)
    y_coords = []
    x_coords = []
    counter = 0
    for t in coords:
        k = int(500 * (t + 1))
        if k == 1000:
            k = k - 1
        x_coords.append(k)

        y_coords.append(y[k])
    if flag == "deep":
        cdot = 'gs'
    else:
        cdot = 'rs'
    # plt.plot(x_coords,y_coords,'bs')
    for counter in range(31999):
        print counter
        fig = plt.figure()
        plt.plot(y)
        ml=mean_loss[0][counter]
        m_l = min_loss[0][counter]
        plt.figtext(.6, .8, "Min_loss =" + str(m_l))
        plt.figtext(.6, .85, "Mean_loss =" + str(ml))
        plt.figtext(.6, .9, "Step = " + str(counter))
        matplotlib.rcParams.update({'font.size': 18})
        fig.set_size_inches(12.8, 12.8)
        X2 = coord_data[counter + 1]
        # print X2
        coords = X2
        # plt.plot(y)
        y_end_coords = []
        x_end_coords = []
        # counter = 0
        for p in coords:

            k = int(500 * (p + 1))
            if k == 1000:
                k = k - 1
            x_end_coords.append(k)

            y_end_coords.append(y[k])
            # print y_end_coords
            # plt.plot(x_coords, y_coords, cdot)
        for i in range(500):
            # print i
            plt.arrow(x_coords[i], y_coords[i], 0, -1 * y_coords[i] + y_end_coords[i], hold=None,
                      head_width=0.01, color=cdot.rstrip('s'))
            plt.arrow(x_coords[i], y_end_coords[i], x_end_coords[i] - x_coords[i], 0, head_width=0.01,
                      color=cdot.rstrip('s'))
        # print counter
        name_out = str(counter) + ".png"
        plt.savefig(os.path.join(out_path, name_out))
        plt.close()


if __name__ == "__main__":
    num_iterations = 32000
    """
    names = ["fb2_mean_loss_[50, 50, 10]fsin0.0016","normalized10_mean_loss_[50, 50, 10]fsin0.0016",
             "fb2_mean_loss_[50, 50, 10]fsin0.0008", "normalized10_mean_loss_[50, 50, 10]fsin0.0008",
             "fb2_mean_loss_[50, 50, 10]fsin0.0004", "normalized10_mean_loss_[50, 50, 10]fsin0.0004",
             "fb2_mean_loss_[50, 50, 10]fsin0.0002", "normalized10_mean_loss_[50, 50, 10]fsin0.0002",
             "fb2_mean_loss_[50, 50, 10]fsin0.0001", "normalized10_mean_loss_[50, 50, 10]fsin0.0001",
             ]
    
    
    names = ["r10norm_2l_min_loss_[50, 50, 10]schwefel7.5e-05","fb2_min_loss_[50, 50, 10]schwefel0.0016",
             "10norm_2l_min_loss_[50, 50, 10]schwefel0.0008", "fb2_min_loss_[50, 50, 10]schwefel0.0008",
             "10norm_2l_min_loss_[50, 50, 10]schwefel0.0004", "fb2_min_loss_[50, 50, 10]schwefel0.0004",
             "10norm_2l_min_loss_[50, 50, 10]schwefel0.0002", "fb2_min_loss_[50, 50, 10]schwefel0.0002",
             "10norm_2l_min_loss_[50, 50, 10]schwefel0.0001", "fb2_min_loss_[50, 50, 10]schwefel0.0001",
             ]
    
    
    names=["128+r_norm_2l_min_loss_[50, 50, 10]schwefel2.5e-05","64+r_norm_2l_min_loss_[50, 50, 10]schwefel2.5e-05",
           "32+r_norm_2l_min_loss_[50, 50, 10]schwefel2.5e-05","16+r_norm_2l_min_loss_[50, 50, 10]schwefel2.5e-05",
           "8+r_norm_2l_min_loss_[50, 50, 10]schwefel2.5e-05","4+r_norm_2l_min_loss_[50, 50, 10]schwefel2.5e-05",
           "2+r_norm_2l_min_loss_[50, 50, 10]schwefel2.5e-05",
           "1+r_norm_2l_min_loss_[50, 50, 10]schwefel2.5e-05","0.5+r_norm_2l_min_loss_[50, 50, 10]schwefel2.5e-05",
           "0.25+r_norm_2l_min_loss_[50, 50, 10]schwefel2.5e-05","0.125*r_norm_2l_min_loss_[50, 50, 10]schwefel2.5e-05",
           "0.0625+r_norm_2l_min_loss_[50, 50, 10]schwefel2.5e-05","0.03125+r_norm_2l_min_loss_[50, 50, 10]schwefel2.5e-05",
           "0.0078125*r_norm_2l_min_loss_[50, 50, 10]schwefel2.5e-05"]
    
    names = ["16.0*r_momentum9_mean_loss_[50, 50, 10]schwefel2.5e-05","8.0*r_momentum9_mean_loss_[50, 50, 10]schwefel2.5e-05",
             "4.0*r_momentum9_mean_loss_[50, 50, 10]schwefel2.5e-05","2.0*r_momentum9_mean_loss_[50, 50, 10]schwefel2.5e-05"
             ,"1.0*r_momentum9_mean_loss_[50, 50, 10]schwefel2.5e-05","0.5*r_momentum9_mean_loss_[50, 50, 10]schwefel2.5e-05",
             "0.25*r_momentum9_mean_loss_[50, 50, 10]schwefel2.5e-05"]
    """
    names = ["coords0[50, 50, 1]_schwefel0.00375","coords1[50, 50, 1]_schwefel0.00375","coords2[50, 50, 1]_schwefel0.00375",
             "coords3[50, 50, 1]_schwefel0.00375","coords4[50, 50, 1]_schwefel0.00375","coords5[50, 50, 1]_schwefel0.00375",
             "coords6[50, 50, 1]_schwefel0.00375","coords7[50, 50, 1]_schwefel0.00375","coords8[50, 50, 1]_schwefel0.00375",
             "coords9[50, 50, 1]_schwefel0.00375","coords10[50, 50, 1]_schwefel0.00375"]

    plot_coords_data(names,num_iterations)
