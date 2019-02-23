import tensorflow as tf
import numpy as np
import time
import math
import os
import sys
import matplotlib
from matplotlib import pyplot as plt


def rescale(x, a, b, c, d):
    """
    Rescales variable from [a, b] to [c, d]
    """
    return c + ((d - c) / (b - a)) * (x - a)

def fsin(x, xmin=-1, xmax=1):
    x = rescale(x, xmin, xmax, -2,2)

    return tf.math.exp(x/10)-tf.math.cos(math.pi*(x-1))/(tf.abs(x-1)+1)

def sin(x, xmin=-1, xmax=1):
    x = rescale(x, xmin, xmax, -2,2)

    return 1-tf.math.cos(math.pi*(x-1))/(tf.abs(x-1)+1)

def ackley(x, xmin=-1, xmax=1):
    """
    https://www.sfu.ca/~ssurjano/ackley.html
    """
    a = 20.0
    b = 0.2
    c = 2 * np.pi
    x = rescale(x, xmin, xmax, -32.768, 32.768)
    term1 = -a * tf.exp(-b * tf.reduce_mean(x**2,axis=1)**0.5)
    term2 = -tf.exp(tf.reduce_mean(tf.cos(c*x),axis=1))
    return term1 + term2 + a + tf.exp(1.0)

def schwefel(x, xmin=-1, xmax=1):
        """
        https://www.sfu.ca/~ssurjano/schwef.html
        """
        x = rescale(x, xmin, xmax, -500, 500)
        d =tf.to_float(tf.shape(x)[1])
        x = tf.sin(tf.abs(x) ** 0.5) * x
        result = 418.9829 * d - tf.reduce_sum(x,axis=1)
        return result
def add_random_noise(w, mean=0.0, stddev=1.0):
    variables_shape = tf.shape(w)
    noise = tf.random_normal(
        variables_shape,
        mean=mean,
        stddev=stddev,
        dtype=tf.float32,
    )
    return w + noise
def multiply_random_noise(w, mean=1.0, stddev=1.0):
    variables_shape = tf.shape(w)
    noise = tf.random_normal(
        variables_shape,
        mean=mean,
        stddev=stddev,
        dtype=tf.float32,
    )
    return w * noise

def te(sess,t,shape,function,lr,k):
    #print t

    W1 = tf.get_variable("W1", shape=[shape[0], shape[1]],
                         initializer=tf.initializers.variance_scaling(scale=2.0, distribution="uniform"),trainable=False)

    W2 = tf.get_variable("W2", shape=[shape[1], shape[2]],
                         initializer=tf.initializers.variance_scaling(scale=2.0/(shape[1]**0.5), distribution="uniform"))

    act = tf.nn.tanh(tf.matmul(W1, W2))
    ans = eval(function)(act)
    c = tf.reduce_min(ans)
    cost = tf.reduce_mean(ans)


    learning_rate = tf.placeholder(tf.float32, shape=[])
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    #opt = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=1.1)
    # train_step = opt.minimize(cost)
    grads, vars = zip(*opt.compute_gradients(cost))
    #grad_w1=grads[0]#*shape[1]*shape[0]



    grad_w2 = grads[0]*shape[1]

    #sttdev = 16.0/(2**k)
    #grad_w2 = add_random_noise(grad_w2,mean=0.0,stddev=sttdev)

    #sttdev = 1 / (2 ** k)
    #grad_w2 = add_random_noise(grad_w2, mean=1.0, stddev=sttdev)

    train_step = opt.apply_gradients(zip([grad_w2], vars))

    sess.run(tf.global_variables_initializer())
    loss_container = []
    coords_container = []
    lc = []
    for i in range(32000):#range(int(2000 * (1.6e-3 / lr))):
        if i%1000==1 : print i
        sess.run(train_step, feed_dict={learning_rate: lr})
        #print i
        #gw=sess.run(noise)

        #print gw
        printed_value, coords, min_loss = sess.run([cost, act, c])
        if math.isnan(printed_value)==True:
            t=t-1
            break
        loss_container.append(printed_value)
        lc.append(min_loss)

        coords_container.append(coords)

    my_dir = "./results"

    np.asarray(coords_container)
    coords_container=np.mean(coords_container,axis=2)
    #np.savetxt(os.path.join(my_dir, 'coords'+'_deep_'+str(k)+str(shape)+'_' + function + str(lr)),coords_container)
    np.savetxt(os.path.join(my_dir, 'coords'+'_deep_'+str(k)+str(shape)+'_' + function + str(lr))+".csv",coords_container, delimiter=",")
    file_name = str(16.0 / (2 ** (k))) + '*r_momentum11_min_loss_' + str(shape) + function + str(lr) + '.txt'
    my_dir = "./results"
    fname = os.path.join(my_dir, file_name)
    with open(fname, 'w') as f:
        for item in lc:
            f.write("%s\n" % item)
    #mean_loss = np.mean(loss_container, axis=0)
    file_name = str(16.0 / (2 ** (k))) + '*r_momentum11_mean_loss_' + str(shape) + function + str(lr) + '.txt'
    my_dir = "./results"
    fname = os.path.join(my_dir, file_name)
    with open(fname, 'w') as f:
        for item in loss_container:
            f.write("%s\n" % item)

    return loss_container, lc,t

def te_clr(sess,t,shape,function,lr,k):
    #print t

    W1 = tf.get_variable("W1", shape=[shape[0], shape[1]],
                         initializer=tf.initializers.variance_scaling(scale=2.0, distribution="uniform"))#,trainable=False)
    W2 = tf.get_variable("W2", shape=[shape[1], shape[2]],
                         initializer=tf.initializers.variance_scaling(scale=2.0/(shape[1]**0.5), distribution="uniform"))

    act = tf.nn.tanh(tf.matmul(W1, W2))
    ans = eval(function)(act)
    c = tf.reduce_min(ans)
    cost = tf.reduce_mean(ans)


    learning_rate = tf.placeholder(tf.float32, shape=[])
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # train_step = opt.minimize(cost)
    grads, vars = zip(*opt.compute_gradients(cost))
    grad_w1=grads[0]#*shape[1]*shape[0]


    grad_w2 = grads[1]*shape[1]

    train_step = opt.apply_gradients(zip([grad_w1,grad_w2], vars))

    sess.run(tf.global_variables_initializer())
    loss_container = []
    coords_container = []
    lc = []
    for i in range(32000):#range(int(2000 * (1.6e-3 / lr))):
        if i%1000==0 and i!=0 and i!=31000 :
            lr_mod = 5e-4
            print lr_mod
        else : lr_mod=lr
        sess.run(train_step, feed_dict={learning_rate: lr_mod})
        #gw=sess.run(grad_w2)
        #print gw
        printed_value, coords, min_loss = sess.run([cost, act, c])
        if math.isnan(printed_value)==True:
            t=t-1
            break
        loss_container.append(printed_value)
        lc.append(min_loss)

        coords_container.append(coords)


    return loss_container, lc,t


def te_mclr(sess,t,shape,function,lr,k):
    #print t

    W1 = tf.get_variable("W1", shape=[shape[0], shape[1]],
                         initializer=tf.initializers.variance_scaling(scale=2.0, distribution="uniform"))

    W2 = tf.get_variable("W2", shape=[shape[1], shape[2]],
                         initializer=tf.initializers.variance_scaling(scale=2.0/(shape[1]**0.5), distribution="uniform"))

    act = tf.nn.tanh(tf.matmul(W1, W2))
    ans = eval(function)(act)
    c = tf.reduce_min(ans)
    cost = tf.reduce_mean(ans)


    learning_rate = tf.placeholder(tf.float32, shape=[])
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    #opt = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=1.1)
    # train_step = opt.minimize(cost)
    grads, vars = zip(*opt.compute_gradients(cost))
    #grad_w1=grads[0]#*shape[1]*shape[0]


    scaling = (ans)/(c)

    grad_w1 = grads[0] *scaling
    #grad_w2 = tf.multiply(grads[0]*shape[1],scaling)
    grad_w2=grads[1]*shape[1]


    #sttdev = 16.0/(2**k)
    #grad_w2 = add_random_noise(grad_w2,mean=0.0,stddev=sttdev)

    #sttdev = 1 / (2 ** k)
    #grad_w2 = add_random_noise(grad_w2, mean=1.0, stddev=sttdev)

    train_step = opt.apply_gradients(zip([grad_w1,grad_w2], vars))

    sess.run(tf.global_variables_initializer())
    loss_container = []
    coords_container = []
    lc = []
    for i in range(32000):#range(int(2000 * (1.6e-3 / lr))):
        if i%1000==1 : print i
        sess.run(train_step, feed_dict={learning_rate: lr})

        #print i
        #gw=sess.run(noise)

        #print gw
        printed_value, coords, min_loss = sess.run([cost, act, c])
        if math.isnan(printed_value)==True:
            t=t-1
            break
        loss_container.append(printed_value)
        lc.append(min_loss)

    return loss_container,lc,t




def te2l_trainable(sess,t,shape,function,lr):
    #print t

    W1 = tf.get_variable("W1", shape=[shape[0], shape[1]],
                         initializer=tf.initializers.variance_scaling(scale=10.0, distribution="uniform"),trainable=False)   #/shape[1]
    W2 = tf.get_variable("W2", shape=[shape[1], shape[2]],
                         initializer=tf.initializers.variance_scaling(scale=10.0, distribution="uniform")) /(shape[1]**0.5)

    act = tf.nn.tanh(tf.matmul(tf.nn.relu(W1), W2))


    ans=eval(function)(act)

    c = tf.reduce_min(ans)
    cost = tf.reduce_mean(ans)

    #lr = 1e-3
    learning_rate = tf.placeholder(tf.float32, shape=[])
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # train_step = opt.minimize(cost)
    grads, vars = zip(*opt.compute_gradients(cost))

    grad_w2 = grads[0]* shape[1]

    train_step = opt.apply_gradients(zip([grad_w2], vars))

    sess.run(tf.global_variables_initializer())
    loss_container = []
    coords_container = []
    lc = []
    flag =0
    for i in range(2000):
    #for i in range(int(2000 * (1.6e-3 / lr))):



        sess.run(train_step, feed_dict={learning_rate: lr})
        printed_value, coords,min_loss = sess.run([cost, act,c])

        loss_container.append(printed_value)
        lc.append(min_loss)

        coords_container.append(coords)

    return loss_container,lc





def te_facebook(sess, t, shape, function, lr):
    print t


    W1 = tf.get_variable("W1", shape=[shape[0], shape[1]],
                         initializer=tf.initializers.variance_scaling(scale=2.0, distribution="uniform"))
    W2 = tf.get_variable("W2", shape=[shape[1], shape[2]],
                         initializer=tf.initializers.variance_scaling(scale=2.0, distribution="uniform"))
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess)
    top_layer = tf.matmul(tf.nn.relu(W1), W2)
    scaling = 1.7159 / tf.maximum(tf.abs(tf.reduce_max(top_layer)), tf.abs(tf.reduce_min(top_layer)))
    scaling_const = sess.run(scaling)
    top_layer = tf.matmul(tf.nn.relu(W1), W2)
    act = tf.nn.tanh(scaling_const * top_layer)


    ans = eval(function)(act)

    c = tf.reduce_min(ans)
    cost = tf.reduce_mean(ans)

    # lr = 1e-3
    learning_rate = tf.placeholder(tf.float32, shape=[])
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # train_step = opt.minimize(cost)
    grads, vars = zip(*opt.compute_gradients(cost))
    grad_w1 = grads[0]
    grad_w2 = grads[1]

    train_step = opt.apply_gradients(zip([grad_w1,grad_w2], vars))

    sess.run(tf.global_variables_initializer())
    loss_container = []
    coords_container = []
    lc = []
    flag = 0
    #for i in range(20000):
    for i in range(int(2000 * (1.6e-3 / lr))):

        sess.run(train_step, feed_dict={learning_rate: lr})
        printed_value, coords, min_loss = sess.run([cost, act, c])

        loss_container.append(printed_value)
        lc.append(min_loss)

        coords_container.append(coords)

    return loss_container, lc


def te_1l_embedding(sess, t, shape, function, lr,k):
    W = tf.get_variable("W", shape=[shape[1], shape[2]],
                         initializer=tf.initializers.variance_scaling(scale=10.0, distribution="uniform"))

    act = tf.nn.tanh(W)



    ans = eval(function)(act)

    c = tf.reduce_min(ans)
    cost = tf.reduce_mean(ans)

    # lr = 1e-3
    learning_rate = tf.placeholder(tf.float32, shape=[])
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # train_step = opt.minimize(cost)
    grads, vars = zip(*opt.compute_gradients(cost))

    grad_w= tf.nn.tanh(grads[0] * shape[0])
    train_step = opt.apply_gradients(zip([grad_w], vars))

    sess.run(tf.global_variables_initializer())
    loss_container = []
    coords_container = []
    lc = []

    for i in range(32000):  # range(int(2000 * (1.6e-3 / lr))):
        if i % 1000 == 1: print i
        sess.run(train_step, feed_dict={learning_rate: lr})
        # print i
        # gw=sess.run(noise)

        # print gw
        printed_value, coords, min_loss = sess.run([cost, act, c])
        if math.isnan(printed_value) == True:
            t = t - 1
            break
        loss_container.append(printed_value)
        lc.append(min_loss)

        coords_container.append(coords)
    my_dir = "./results"

    np.asarray(coords_container)
    coords_container = np.mean(coords_container, axis=2)
    # np.savetxt(os.path.join(my_dir, 'coords'+'_deep_'+str(k)+str(shape)+'_' + function + str(lr)),coords_container)
    np.savetxt(os.path.join(my_dir, 'coords'  + str(t) + str(shape) + '_' + function + str(lr)) + ".csv",
               coords_container, delimiter=",")
    return loss_container, lc,t


def run(shape):
    shape_=shape
    lr=15e-5


    for j in range(1):

     lr=lr/2
     for k in range(1):
        f_list = ['schwefel']
        for function in f_list:
            mean_loss = []
            min_loss = []
            for t in range(11):
                flag=t
                sess = tf.Session()
                m_l, min_l,t = te_mclr(sess, t, shape_, function,lr,k)
                if flag == t:
                    min_loss.append(min_l)
                    mean_loss.append(m_l)


                sess.close()
                tf.reset_default_graph()
            np.asarray(mean_loss)
            np.asarray(min_loss)
            min_loss = np.mean(min_loss, axis=0)
            file_name = str(16.0/(2**(k)))+'*r_mclr_min_loss_' + str(shape_) + function + str(lr)+'.txt'
            my_dir = "./results"
            fname = os.path.join(my_dir, file_name)
            with open(fname, 'w') as f:
                for item in min_loss:
                    f.write("%s\n" % item)
            mean_loss = np.mean(mean_loss, axis=0)
            file_name = str(16.0/(2**(k)))+'*r_mclr_mean_loss_' + str(shape_) + function + str(lr)+'.txt'
            my_dir = "./results"
            fname = os.path.join(my_dir, file_name)
            with open(fname, 'w') as f:
                for item in mean_loss:
                    f.write("%s\n" % item)



if __name__ == '__main__':
    shape = [50, 50, 1]
    if len(sys.argv) >= 2:
        shape[0] = int(sys.argv[1])
        shape[1] = int(sys.argv[1])
    if len(sys.argv) >= 3:
        shape[2] = int(sys.argv[2])


    run(shape)


