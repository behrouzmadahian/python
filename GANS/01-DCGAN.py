import numpy as np
import tensorflow as tf
import os, time, itertools, imageio, pickle
from matplotlib import  pyplot as plt
from tensorflow.examples.tutorials.mnist import  input_data
'''
we resize the images into (1,64,64,1).
Think about convolution, getting a p*p image (1, 64, 64, 1) and after several convolutions,
reshapes its input to (1,1,1,1). 
now transpose convolution layers of generator get the (1,1,1,100) noise and turn in into (1, 64,64,1),

For the SAME padding, the output height and width are computed as:
out_height = ceil(float(in_height) / float(strides[1]))

out_width = ceil(float(in_width) / float(strides[2]))

And

For the VALID padding, the output height and width are computed as:
out_height = ceil(float(in_height - filter_height) / float(strides[1])) +1

out_width = ceil(float(in_width - filter_width) / float(strides[2])) + 1
'''
# dataset normalization: range -1 -> 1: (pixelVal -0.5)/0.5

# training parameters:
batchSize = 100
lr = 0.0002
training_epoch = 20
optimizer = tf.train.AdamOptimizer
dropout = 0.3

def lrelu(x, th =0.2):
    return tf.maximum(th*x, x)
#Generator: G(z):
def generator(x, isTrain = True, reuse =False):
    # x is some sort of a noise..
    with tf.variable_scope('generator', reuse =reuse):
        # 1st hidden layer
        conv1 = tf.layers.conv2d_transpose(x, 1024, [4, 4], strides=(1, 1), padding='valid')
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

        # 2nd hidden layer:
        conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd hidden layer:
        conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # 4th conv layer:
        conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

        # output layer:
        conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same')
        o = tf.nn.tanh(conv5)
        print('Generator: Shape of output of each transpose convolution layer:')
        print(x.get_shape(),conv1.get_shape(), conv2.get_shape(), conv3.get_shape(), conv4.get_shape(), conv5.get_shape())
        return o

def discriminator(x, isTrain =True, reuse =False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # 1st hidden layer
        conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu1 = lrelu(conv1, 0.2)

        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # 4th hidden layer
        conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding='same')
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

        # output layer
        conv5 = tf.layers.conv2d(lrelu4, 1, [4, 4], strides=(1, 1), padding='valid')
        o = tf.nn.sigmoid(conv5)
        print('Discriminator: Shape of output of each convolution layers:')
        print(x.get_shape(),conv1.get_shape(), conv2.get_shape(), conv3.get_shape(), conv4.get_shape(), conv5.get_shape())
        return o, conv5

fixed_Z = np.random.normal(0, 1, (25, 1, 1, 100))
x = tf.placeholder(tf.float32, shape = (None, 64,64,1))
z = tf.placeholder(tf.float32, shape = (None, 1, 1, 100))
isTrain = tf.placeholder(dtype=tf.bool)
#networks: generator:
G_z = generator(z, isTrain=isTrain)
print('Shape of output of Generator:', G_z.get_shape())

# discriminator network:
d_real, d_real_logits = discriminator(x, isTrain)
d_fake, d_fake_logits = discriminator(G_z, isTrain, reuse =True)
##
# Measures the probability error in discrete classification tasks
#  in which each class is independent and not mutually exclusive.
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits,
                                                                     labels= tf.ones([batchSize, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits,
                                                                     labels=tf.zeros([batchSize, 1, 1, 1])))
D_loss = d_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits,
                                                                labels=tf.ones([batchSize, 1, 1, 1])))
# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
# BATCH NORMALIZATION HAS EXTRA  parameters that wee need to trigger extra_update_ops
# to do so we use control dependeincies on the optimizers as follows:
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)


def show_result(sess, num_epoch, show = False, path = 'results.png', isFix = False):
    z_ = np.random.normal(0, 1, (25,1,1, 100))
    if isFix:
        test_images = sess.run(G_z, feed_dict = {z: fixed_Z, isTrain :False})
    else:
        test_images = sess.run(G_z, feed_dict = {z: z_, isTrain: False})
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize = (5,5))
    for i,j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i,j].get_yaxis().set_visible(False)
    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i,j].cla() #clears an axis, i.e. the currently active axis in the current figure. It leaves the other axes untouched.
        ax[i,j].imshow(np.reshape(test_images[k], (64, 64)), cmap ='gray')

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
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True, reshape =[]) # reshape=[]: returns (N, 28, 28, 1)
print('Shape of image data before resizing:', mnist.train.images[0].shape)

# loss for each network:
init = tf.global_variables_initializer()

# results save folder
# results save folder
root = 'MNIST_DCGAN_results/'
model = 'MNIST_DCGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')
if not os.path.isdir(root+ 'Random_results'):
    os.mkdir(root+'Random_results')
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

with tf.Session() as sess:
    sess.run(init)
    # MNIST resize and normalization
    train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval()
    train_set = (train_set - 0.5) / 0.5
    np.random.seed(int(time.time()))
    start_time = time.time()
    for epoch in range(training_epoch):
        p = root + 'random_Results/' + model + str(epoch) + '.png'
        fixed_p = root + 'Fixed_results/' + model + str(epoch) + '.png'
        show_result(sess, (epoch + 1), show=False, path=p, isFix=False)
        show_result(sess, (epoch + 1), show=False, path=fixed_p, isFix=True)
        G_losses = []
        D_losses = []
        epoch_start_time = time.time()
        # update discriminator 3 times more often!
        kk = 1
        for iter in range(train_set.shape[0]//batchSize):
            # update discriminator:
            x_ = train_set[iter*batchSize:(iter+1)*batchSize]
            print(x_.shape)
            z_ = np.random.normal(0, 1, (batchSize,1,1, 100))
            loss_d_, _ = sess.run([D_loss, D_optim], feed_dict={x:x_, z:z_, isTrain:True})
            D_losses.append(loss_d_)

            # update generator
            if epoch < 10:
                if kk % 2 == 0:
                    z_ = np.random.normal(0, 1, (batchSize,1,1, 100))
                    loss_g_, _ = sess.run([G_loss, G_optim], feed_dict={z: z_,x:x_, isTrain:True})
                    G_losses.append(loss_g_)
                kk += 1

            else:
                z_ = np.random.normal(0, 1, (batchSize,1,1, 100))
                loss_g_, _ = sess.run([G_loss, G_optim], feed_dict={z: z_, x:x_, isTrain:True})
                G_losses.append(loss_g_)

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), training_epoch, per_epoch_ptime,
                                                                    np.mean(D_losses), np.mean(G_losses)))


        train_hist['D_losses'].append(np.mean(D_losses))
        train_hist['G_losses'].append(np.mean(G_losses))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']),
                                                                  training_epoch, total_ptime))
print("Training finish!... save training results")
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)
show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

images = []
for e in range(training_epoch):
    img_name = root + model +'Fixed_results/'+model+  str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + model +'generation_animation.gif', images, fps=5)

sess.close()
