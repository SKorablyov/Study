import tensorflow as tf
import math
import numpy as np
import os
import sys
import itertools
#import conf




def loss_multi_dot(act,function,shape,num_points):
    for i in range(num_points-1):
        if i ==0:

            w_sub = act[i]
            l = eval(function)(w_sub)
            l = tf.expand_dims(input=l,axis=0)
        #print(i)
        w_sub = act[i]
        loss = eval(function)(w_sub)
        loss = tf.expand_dims(input=loss,axis=0)
        l = tf.concat([l,loss],axis=0)
    return l


def generate_distribution_ns(number_of_points,dim):
    output = np.zeros([number_of_points,dim,1])
    if dim == 1 :
        for i in range(number_of_points):
            output[i][0][0] = i * 1/(number_of_points)
    elif dim == 2 :
        k = 0.0
        for i in range(number_of_points):
            output[i][0][0] = (1.0/31)*(i%32)
            output[i][1][0] = k
            if i % 32 == 31 :
                k += 1.0/31

    elif dim == 4:
        k = (0,0.33,0.66,1)
        iterables = [k,k,k,k]
        j = []
        counter = 0
        for t in itertools.product(*iterables):
            j.append(t)
        output = np.asarray(j)
        output = np.expand_dims(output,axis = 2)
        print(counter)

    elif dim == 8:
        k = (0,1)
        iterables = [k, k, k, k,k,k,k,k]
        j = []
        counter = 0
        for t in itertools.product(*iterables):
            j.append(t)
        output = np.asarray(j)
        output = np.expand_dims(output, axis=2)
        print(counter)
    else :
        output = np.random.uniform(0.0,1.0,(number_of_points ,dim,1))
    return output


def rescale(x, a, b, c, d):
    """
    Rescales variable from [a, b] to [c, d]
    """
    return c + ((d - c) / (b - a)) * (x - a)

def fsin(x, xmin=-1, xmax=1):
    x = rescale(x, xmin, xmax, -2,2)

    return tf.math.exp(x/10)-tf.math.cos(math.pi*(x-1))/(tf.abs(x-1)+1)


def csin(x, xmin=-1, xmax=1):
    x = rescale(x, xmin, xmax, 0,2)

    return tf.math.sin(2*(x+0.25)*math.pi)+tf.math.cos(x*math.pi)+x
def rcsin(x, xmin=-1, xmax=1):
    x = rescale(x, xmin, xmax, 0,2)

    return tf.math.sin(2*(x+0.25)*math.pi)+tf.math.cos(x*math.pi)-x+2
def sin(x, xmin=-1, xmax=1):
    x = rescale(x, xmin, xmax, -2,2)

    return 1-tf.math.cos(math.pi*(x-1))/(tf.abs(x-1)+1)

def schwefel(x, xmin=-1, xmax=1):
        """
        https://www.sfu.ca/~ssurjano/schwef.html
        """
        x = rescale(x, xmin, xmax, -500, 500)
        d = tf.to_float(tf.shape(x)[1])
        x = tf.sin(tf.abs(x) ** 0.5) * x
        result = 418.9829 * d - tf.reduce_sum(x, axis=1)
        return result


def te(sess,t,shape,function,lr,k,w0,number_of_points,name):

    #name='shatai'


    y=1
    flag=tf.get_variable("flag",shape=[1],initializer=tf.zeros_initializer(),trainable=False)

    W0 = tf.get_variable("W0", shape=[1,shape[2], shape[0]],
                         initializer=tf.initializers.variance_scaling(scale=2.0, distribution="uniform"),trainable=False,dtype=tf.float64)

    #w=tf.random_normal(shape=[1000,shape[2],1],mean=0,stddev=0.05)

    #W = tf.matmul(w0,W0)
    W=w0*W0
    #print(W)
    W1 = tf.get_variable("W1", shape=[shape[0], shape[1]],
                        initializer=tf.initializers.variance_scaling(scale=2.0, distribution="uniform"),trainable=False,dtype=tf.float64)

    W1 = tf.contrib.framework.sort(W1)
    W11 = tf.expand_dims(W1,axis=0)
    W11 = tf.tile(W11,[number_of_points,1,1])
    W=tf.matmul(W,W11)
    #W=W*W11

    W2 = tf.get_variable("W2", shape=[number_of_points,shape[1], shape[2]],
                         initializer=tf.initializers.variance_scaling(scale=(2.0/shape[1]**0.5), distribution="uniform"),dtype=tf.float64)

    W20_1 = tf.reshape(W2, shape=[number_of_points * shape[1] * shape[2]])


    act = tf.nn.tanh(tf.matmul(W, W2))
    ans = eval(function)(act)
    print(ans)
    t_ans = tf.reduce_mean(ans,axis=1)
    c = tf.reduce_min(t_ans)
    cost = tf.reduce_mean(t_ans)


    learning_rate = tf.placeholder(tf.float32, shape=[])
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    grads, vars = zip(*opt.compute_gradients(cost))




    grad_w2 = grads[0]

    train_step = opt.apply_gradients(zip([grad_w2], vars))

    sess.run(tf.global_variables_initializer())
    loss_container = []
    coords_container = []
    lc = []
    ans_cont=[]
    for i in range(10000):#range(int(2000 * (1.6e-3 / lr))):

        sess.run(train_step, feed_dict={learning_rate: lr})
        #print i
        #gw=sess.run(noise)
        if i%5000==4999 :
            y+=0
            #flag=flag+1
        if i%1000==1 :

            print ('Step=',i,shape)
            print('Mean loss = ',printed_value)
            print('Min loss =',min_loss)
            #print(sess.run(flag))

            X0 = sess.run(W20_1)
            # print(X0)
            coords_container.append(X0)

            t_ans_sess = sess.run(tf.reshape(t_ans, [number_of_points*shape[2]]))
            # print(t_ans)
            ans_cont.append(t_ans_sess)
        #print gw
        printed_value, coords, min_loss = sess.run([cost, act, c])
        if math.isnan(printed_value)==True:
            t=t-1
            break
        loss_container.append(printed_value)
        lc.append(min_loss)




    my_dir = "./results"
    #np.savetxt(os.path.join(my_dir, 'coords'+'_deep_'+str(k)+str(shape)+'_' + function + str(lr)),coords_container)
    #np.savetxt(os.path.join(my_dir, 'coords'+'_deep_'+str(k)+str(shape)+'_' + function + str(lr))+".csv",coords_container, delimiter=",")
    
    file_name = name+'_min_loss_' + str(shape) + function  + '.txt'
   
    fname = os.path.join(my_dir, file_name)
    with open(fname, 'w') as f:
        for item in lc:
            f.write("%s\n" % item)
    #mean_loss = np.mean(loss_container, axis=0)
    file_name =name + 'mean_loss_' + str(shape) + function + '.txt'
    my_dir = "./results"
    fname = os.path.join(my_dir, file_name)
    with open(fname, 'w') as f:
        for item in loss_container:
            f.write("%s\n" % item)



    my_dir = './results'
    """
    np.asarray(coords_container)
    np.savetxt(os.path.join(my_dir, name +'n_weight' + str(shape) + '_' + function + str(lr)) + ".csv",
           coords_container, delimiter=",")
    
    np.asarray(ans_cont)
    np.savetxt(os.path.join(my_dir,name+ 'n_num' + str(shape) + '_' + function + str(lr)) + ".csv",
           ans_cont, delimiter=",")
    """
    w_ = sess.run(W0)
    np.save(os.path.join(my_dir, name + 'w0' + str(shape) + '_' + function + str(lr)) + ".npy",
            w_)

    w_1 = sess.run(W1)
    np.save(os.path.join(my_dir, name + 'w1' + str(shape) + '_' + function + str(lr)) + ".npy",
            w_1, )

    w_2 = sess.run(W2)
    np.save(os.path.join(my_dir, name + 'w2' + str(shape) + '_' + function + str(lr)) + ".npy",
            w_2)

    return loss_container, lc,t


if __name__ == '__main__':


    #shape = [1,1,1]
    #shape = [16, 16, 1],[1,1,2],[2,2,2],[4, 4, 2],[8,8,2],[16, 16, 2]
    #shape = [2, 2, 1],[1,1,4],[2,2,4],[4, 4, 4],[8,8,4],[16, 16, 4],[1,1,8],[2,2,8],[4, 4, 8],[8,8,8],[16, 16, 8]
    #shape = [4, 4, 1],[32,32,1],[64,64,1],[128,128,1],[256,256,1]


    #iterables = [[1,2,3,4],[1,2,3,4]]


    shapes = [[1,1,1],[2,2,1],[4, 4, 1],[8,8,1],[16, 16, 1]]


    sh =16
    number_of_points = 1024
    if sh == 4 or sh == 8:
        number_of_points = 256


    #shapes =[[1,1,1],[2, 2, 1],[4, 4, 1],[8,8,1]]

    if len(sys.argv) >= 2:
        sh = int(sys.argv[1])

    t = 0
    functions = ['rcsin','fsin','sin']
    k = 0
    my_dir = './results/'
    name = 'experiment'
    w0 =generate_distribution_ns(number_of_points,sh)
    np.save(my_dir + name + 'input' + '_' + str(sh) + ".npy", w0)
    w_0  = np.reshape(w0,(number_of_points ,sh*1))


    np.savetxt(os.path.join(my_dir, name + 'input' + '_' +str(sh) + ".csv" ),w_0, delimiter=",")
    for sh_l in range(7):
        sh = 2**sh_l

        for function in functions:
            #for f_l in range(7):
                for shape in shapes:
                    shape[2] = shape[2] * sh
                    #shape[0] = 2 ** f_l

                    sess = tf.Session()

                    # lr=7.5e-3
                    lr = 0.5 * 7.5e-5 * number_of_points * number_of_points * (shape[2] ** 2)
                    #
                    lr = round(lr / shape[0] ** 0.5, 8)
                    te(sess, t, shape, function, lr, k, w0, number_of_points, name)
                    sess.close()
                    tf.reset_default_graph()
                    #
                    lr = lr * shape[0] ** 0.5
                    shape[2] = int(shape[2] / sh)
            #f_l=0

