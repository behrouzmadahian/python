import numpy as np
import tensorflow as tf
import os, time, itertools, imageio, pickle
from matplotlib import  pyplot as plt
from tensorflow.examples.tutorials.mnist import  input_data
# dataset normalization: range -1 -> 1: (pixelVal -0.5)/0.5

# fully connected Model
# training parameters:
batchSize = 500
lr = 0.0002
training_epoch = 100
optimizer = tf.train.AdamOptimizer
dropout = 0.3

#Generator: G(z):
def generator(x, reuse=False):
    with tf.variable_scope('generator', reuse =reuse):
        # x is some sort of a noise..
        # initializers:
        w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
        b_init = tf.constant_initializer(0.)
        # 1st hidden layer:
        w0 = tf.get_variable('G_w0', shape=[x.get_shape()[1], 256], initializer=w_init)
        b0 = tf.get_variable('G_b0', shape=[256], initializer=b_init)
        h0 = tf.nn.relu(tf.matmul(x, w0) + b0)

        # 2nd hidden layer:
        w1 = tf.get_variable('G_w1', shape=[h0.get_shape()[1], 512], initializer=w_init)
        b1 = tf.get_variable('G_b1', shape=[512], initializer=b_init)
        h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)

        # 3rd hidden layer:
        w2 = tf.get_variable('G_w2', shape=[h1.get_shape()[1], 1024], initializer=w_init)
        b2 = tf.get_variable('G_b2', shape=[1024], initializer=b_init)
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

        # output layer:
        w3 = tf.get_variable('G_w3', shape=[h2.get_shape()[1], 784], initializer=w_init)
        b3 = tf.get_variable('G_b3', shape=[784], initializer=b_init)
        o = tf.nn.tanh(tf.matmul(h2, w3) + b3)
        return o



# Discriminator D(x)
def discriminator(x, drop_out, reuse=False):
    with tf.variable_scope('discriminator', reuse =reuse):
        # initializers:
        w_init = tf.truncated_normal_initializer(mean=0, stddev=0.02)
        b_init = tf.constant_initializer(0.)
        # 1st hidden layer:
        w0 = tf.get_variable('D_w0', shape=[x.get_shape()[1], 1024], initializer=w_init)
        b0 = tf.get_variable('D_b0', shape=[1024], initializer=b_init)
        h0 = tf.nn.relu(tf.matmul(x, w0) + b0)
        h0 = tf.nn.dropout(h0, keep_prob=drop_out)

        # 2nd hidden layer:
        w1 = tf.get_variable('D_w1', shape=[h0.get_shape()[1], 512], initializer=w_init)
        b1 = tf.get_variable('D_b1', shape=[512], initializer=b_init)
        h1 = tf.nn.relu(tf.matmul(h0, w1) + b1)
        h1 = tf.nn.dropout(h1, keep_prob=drop_out)

        # 3rd hidden layer
        w2 = tf.get_variable('D_w2', shape=[h1.get_shape()[1], 256], initializer=w_init)
        b2 = tf.get_variable('D_b2', shape=[256], initializer=b_init)
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
        h2 = tf.nn.dropout(h2, keep_prob=drop_out)

        # output layer:
        w3 = tf.get_variable('D_w3', shape=[h2.get_shape()[1], 1], initializer=w_init)
        b3 = tf.get_variable('D_b3', shape=[1], initializer=b_init)
        o = tf.nn.sigmoid(tf.matmul(h2, w3) + b3)
        return o

fixed_Z = np.random.normal(0, 1, (25, 100))
drop_out = tf.placeholder(tf.float32, name = 'drop_out')
x = tf.placeholder(tf.float32, shape = (None, 784))
z = tf.placeholder(tf.float32, shape = (None, 100))

# Generator network:
G_z = generator(z)

# discriminator network:
d_real = discriminator(x, drop_out)
d_fake = discriminator(G_z, drop_out, reuse =True)

def show_result(sess, num_epoch, show = False, path = 'results.png', isFix = False):
    z_ = np.random.normal(0, 1, (25, 100))
    if isFix:
        test_images = sess.run(G_z, feed_dict = {z: fixed_Z, drop_out: 1.0})
    else:
        test_images = sess.run(G_z, feed_dict = {z: z_, drop_out: 1.0})
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize = (5,5))
    for i,j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i,j].get_yaxis().set_visible(False)
    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i,j].cla() #clears an axis, i.e. the currently active axis in the current figure. It leaves the other axes untouched.
        ax[i,j].imshow(np.reshape(test_images[k], (28, 28)), cmap ='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha = 'center')
    plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save =False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))
    y1 = hist['D_losses']
    y2 = hist['G_losses']
    plt.plot(x, y1, label ='D_losses')
    plt.plot(x, y2, label ='G_losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()

# load mnist:
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
train_set = (mnist.train.images -0.5)/0.5

# loss for each network:
eps = 1e-2
d_loss = tf.reduce_mean(-tf.log(d_real+eps) - tf.log(1-d_fake+ eps))
g_loss = tf.reduce_mean(-tf.log(d_fake+eps))

# trainable variables for each network:
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if  var.name.startswith('discriminator')]
g_vars = [var for var in t_vars if var.name.startswith('generator')]

# optimizer for each network:
d_optim = tf.train.AdamOptimizer(lr).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(lr).minimize(g_loss, var_list=g_vars)

init = tf.global_variables_initializer()

# results save folder
if not os.path.isdir('MNIST_GAN_results'):
    os.mkdir('MNIST_GAN_results')
if not os.path.isdir('MNIST_GAN_results/Random_results'):
    os.mkdir('MNIST_GAN_results/Random_results')
if not os.path.isdir('MNIST_GAN_results/Fixed_results'):
    os.mkdir('MNIST_GAN_results/Fixed_results')
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

with tf.Session() as sess:
    sess.run(init)
    np.random.seed(int(time.time()))
    start_time = time.time()
    for epoch in range(training_epoch):
        G_losses = []
        D_losses = []
        epoch_start_time = time.time()
        # update discriminator 3 times more often!
        kk = 1
        for iter in range(train_set.shape[0]//batchSize):
            # update discriminator:
            x_ = train_set[iter*batchSize:(iter+1)*batchSize]
            z_ = np.random.normal(0, 1, (batchSize, 100))
            loss_d_, _ = sess.run([d_loss, d_optim], feed_dict={x:x_, z:z_, drop_out:0.3})
            D_losses.append(loss_d_)

            # update generator
            if epoch < 10:
                if kk % 2 == 0:
                    z_ = np.random.normal(0, 1, (batchSize, 100))
                    loss_g_, _ = sess.run([g_loss, g_optim], feed_dict={z: z_, drop_out: 0.3})
                    G_losses.append(loss_g_)
                kk += 1

            else:
                z_ = np.random.normal(0, 1, (batchSize, 100))
                loss_g_, _ = sess.run([g_loss, g_optim], feed_dict={z: z_, drop_out: 0.3})
                G_losses.append(loss_g_)

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), training_epoch, per_epoch_ptime,
                                                                    np.mean(D_losses), np.mean(G_losses)))

        p = 'MNIST_GAN_results/Random_results/MNIST_GAN_' + str(epoch + 1) + '.png'
        fixed_p = 'MNIST_GAN_results/Fixed_results/MNIST_GAN_' + str(epoch + 1) + '.png'
        show_result(sess,(epoch + 1), show=False,  path=p, isFix=False)
        show_result(sess, (epoch + 1), show=False, path=fixed_p, isFix=True)
        train_hist['D_losses'].append(np.mean(D_losses))
        train_hist['G_losses'].append(np.mean(G_losses))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']),
                                                                  training_epoch, total_ptime))
print("Training finish!... save training results")
with open('MNIST_GAN_results/train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)
show_train_hist(train_hist, save=True, path='MNIST_GAN_results/MNIST_GAN_train_hist.png')

images = []
for e in range(training_epoch):
    img_name = 'MNIST_GAN_results/Fixed_results/MNIST_GAN_' + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave('MNIST_GAN_results/generation_animation.gif', images, fps=5)

sess.close()
