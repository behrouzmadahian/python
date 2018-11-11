import numpy as np
import tensorflow as tf
import os, time, itertools, imageio, pickle
from matplotlib import  pyplot as plt
from tensorflow.examples.tutorials.mnist import  input_data
import tensorflow.contrib.slim as slim
'''
Here the following stuipulations have been made:
1- there are only labeled data and generated images and no unlabeled data.
2- the regular GAN loss is optimized <NO feature matching loss for Generator>
##############
3- one-sided label smoothing : make sure this happens properly.
In the paper, for the real and unlabled data they use 0.9 instead of 1 as label in calculating 
-0.9*log(P(Real)) =  -0.9 * log(1-P(fake)); p(fake)= p(y=k+1)
However, here this happened on fake lables! and cross entropy across all labels! 
In version 2 of the code I will do this as mentioned in the paper!
##############
Discriminator Loss:
d_loss = supervised Loss for real data + GAN sample loss < -log(P(D(G(z))=k+1) >
Generator loss:
g_loss = -log(P(D(G(z)) =k+1) + Huber_loss(real_image, fake_image)*some weight<1!! : This is not in the original paper
the Buber_loss only exists for 1500 batches!!!

we reshape the images into (1,28,28,1).
batch normalization MUST BE APPLIED before activation!!

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
batch_size = 100
lr = 0.0001
training_epoch = 50
optimizer = tf.train.AdamOptimizer
n_classes = 10
drop_rate = 0.2

def lrelu(x, th =0.2):
    return tf.maximum(th*x, x)

def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)
#Generator: G(z):
def generator(x, isTrain, reuse =False):
    # x is some sort of a noise..
    with tf.variable_scope('generator', reuse =reuse):
        # 1st hidden layer
        conv1 = tf.layers.conv2d_transpose(x, 100, [2, 2], strides=(1, 1), padding='valid')
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

        # 2nd hidden layer:
        conv2 = tf.layers.conv2d_transpose(lrelu1, 25, [3, 3], strides=(2, 2), padding='valid')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd hidden layer:
        conv3 = tf.layers.conv2d_transpose(lrelu2, 6, [4, 4], strides=(2, 2), padding='valid')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # 4th conv layer:
        conv4 = tf.layers.conv2d_transpose(lrelu3, 1, [6, 6], strides=(2, 2), padding='valid')
        # output layer:
        o = tf.nn.tanh(conv4)
        print('Generator: Shape of output of each transpose convolution layer:')
        print(x.get_shape(),conv1.get_shape(), conv2.get_shape(), conv3.get_shape(), conv4.get_shape())
        return o

def discriminator(x, isTrain, reuse =False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # 1st hidden layer
        conv1 = tf.layers.conv2d(x, 32, [4, 4], strides=(2, 2), padding='same')
        lrelu1 = lrelu(conv1, 0.2)
        lrelu1 = tf.layers.dropout(lrelu1, rate=drop_rate, training=isTrain)
        # 2nd hidden layer
        conv2 = tf.layers.conv2d(lrelu1, 64, [4, 4], strides=(2, 2), padding='same')
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)
        lrelu2 = tf.layers.dropout(lrelu2, rate=drop_rate, training=isTrain)

        # 3rd hidden layer
        conv3 = tf.layers.conv2d(lrelu2, 128, [4, 4], strides=(2, 2), padding='same')
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)
        lrelu3 = tf.layers.dropout(lrelu3, rate=drop_rate, training=isTrain)

        flat_dim = np.prod(lrelu3.get_shape()[1:])

        lrelu3_flat = tf.reshape(lrelu3, [-1, flat_dim])
        # Fully connected output layer
        out = tf.layers.dense(lrelu3_flat, n_classes+1, activation=None)
        print('Discriminator: Shape of output of each convolution layers:')
        print(x.get_shape(),conv1.get_shape(), conv2.get_shape(), conv3.get_shape(),out.get_shape())#, out.get_shape)
        return tf.nn.softmax(out), out

fixed_Z = np.random.normal(0, 1, (25, 1, 1, 100))
x = tf.placeholder(tf.float32, shape = (None, 28,28,1))
label = tf.placeholder(tf.float32, shape=[None, n_classes])
z = tf.placeholder(tf.float32, shape = (None, 1, 1, 100))
isTrain = tf.placeholder(dtype=tf.bool)
recon_weight = tf.placeholder_with_default(tf.cast(1.0, tf.float32), shape=[])
#networks: generator:
G_z = generator(z, isTrain=isTrain)
fake_image = G_z
print('Shape of output of Generator:', G_z.get_shape())

# discriminator network:
d_real, d_real_logits = discriminator(x, isTrain)
d_fake, d_fake_logits = discriminator(G_z, isTrain, reuse =True)
##
# Bulding the loss functions:
def build_loss(d_real, d_real_logits, d_fake, d_fake_logits, label, real_image, fake_image, recon_weight):
    # d_real: softmax output [B, n+1], d_real_logits: [B, n+1]
    alpha = 0.9
    real_label = tf.concat([label, tf.zeros((batch_size, 1))], axis=1)  # make it n+1 classes!
    fake_label = tf.concat([(1 - alpha) * tf.ones((batch_size, n_classes)) / n_classes, alpha * tf.ones([batch_size, 1])],
                           axis=1)
    # Discriminator/ classifier loss:
    s_loss = tf.reduce_mean(huber_loss(label, d_real[:, :-1]))
    d_loss_real = tf.nn.softmax_cross_entropy_with_logits(logits=d_real_logits, labels=real_label)
    d_loss_fake = tf.nn.softmax_cross_entropy_with_logits(logits=d_fake_logits, labels=fake_label)

    d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)
    # generator loss:
    g_loss = tf.reduce_mean(tf.log(d_fake[:, -1]))  # log(P(D(G(z)))
    # weight annealing
    g_loss += tf.reduce_mean(huber_loss(real_image, fake_image)) * recon_weight
    GAN_loss = tf.reduce_mean(d_loss + g_loss)

    # classification accuracy:
    correct_prediction = tf.equal(tf.argmax(d_real[:, :-1], axis=1), tf.argmax(label, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return s_loss, d_loss_real, d_loss_fake, d_loss, g_loss, GAN_loss, accuracy


s_loss, d_loss_real, d_loss_fake, d_loss, g_loss, GAN_loss, accuracy =build_loss(d_real, d_real_logits, d_fake, d_fake_logits,
                                                                                 label, x, fake_image, recon_weight)
# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
learning_rate = tf.train.exponential_decay(lr,   global_step=global_step,
                                           decay_steps=10000, decay_rate=0.5,
                                           staircase=True, name='decaying_learning_rate')

# optimizer for each network
# BATCH NORMALIZATION HAS EXTRA  parameters that wee need to trigger extra_update_ops
# to do so we use control dependeincies on the optimizers as follows:
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    d_optimizer = tf.contrib.layers.optimize_loss(loss=d_loss,
                                                  global_step=global_step,
                                                  learning_rate= learning_rate,
                                                  optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                  clip_gradients=20.0,
                                                  name='d_optimize_loss',
                                                  variables=D_vars)

    g_optimizer = tf.contrib.layers.optimize_loss(loss=g_loss,
                                                  global_step=global_step,
                                                  learning_rate=learning_rate,
                                                  optimizer=tf.train.AdamOptimizer(beta1=0.5),
                                                  clip_gradients=20.0,
                                                  name='g_optimize_loss',
                                                  variables=G_vars)

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
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True, reshape =[]) # reshape=[]: returns (N, 28, 28, 1)
print('Shape of image data before resizing:', mnist.train.images[0].shape)

# loss for each network:
init = tf.global_variables_initializer()

# results save folder
# results save folder
root = 'C:/behrouz/Research_and_Development/GANS/codes/SSGAN/results/'
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
    #train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval()
    train_set, trainY = mnist.train.images, mnist.train.labels
    train_set = (train_set - 0.5) / 0.5
    test_set, testY = mnist.test.images, mnist.test.labels
    test_set = (test_set - 0.5) / 0.5
    np.random.seed(int(time.time()))
    start_time = time.time()
    inds = np.arange(len(train_set))
    step =0
    maxIter = (train_set.shape[0] // batch_size) * training_epoch
    for epoch in range(training_epoch):
        acc = sess.run(accuracy, feed_dict={x:train_set,label:trainY, isTrain:False})
        test_acc = sess.run(accuracy, feed_dict={x:test_set,label:testY, isTrain:False})
        print('Epoch= ', epoch, 'Train Accuracy=', acc, 'Test Accuracy= ', test_acc)
        p = root + 'random_Results/' +  str(epoch) + '.png'
        fixed_p = root + 'Fixed_results/' +  str(epoch) + '.png'
        show_result(sess, (epoch + 1), show=False, path=p, isFix=False)
        show_result(sess, (epoch + 1), show=False, path=fixed_p, isFix=True)
        G_losses = []
        D_losses = []
        epoch_start_time = time.time()
        # update discriminator 3 times more often!
        kk = 1
        np.random.shuffle(inds)
        train_set = train_set[inds]
        trainY = trainY[inds]
        for iter in range(train_set.shape[0]//batch_size):
            step += 1
            # update discriminator:
            x_ = train_set[iter*batch_size:(iter+1)*batch_size]
            y_ = trainY [iter*batch_size:(iter+1)*batch_size]
            z_ = sess.run(tf.random_uniform([batch_size,1,1, 100], minval = -1, maxval =1, dtype = tf.float32))

            loss_d_, _ = sess.run([d_loss, d_optimizer], feed_dict={x:x_,label:y_, z:z_, isTrain:True})
            D_losses.append(loss_d_)
            anneal_weight = min(max(0, (1500 - step) / 1500), 1.0)
            # update generator
            if epoch < 10:
                if kk % 2 == 0:
                   # z_ = np.random.normal(0, 1, (batch_size,1,1, 100))
                    loss_g_, _ = sess.run([g_loss, g_optimizer], feed_dict={z: z_,x:x_,recon_weight:anneal_weight, isTrain:True})
                    G_losses.append(loss_g_)
                kk += 1
            else:
                #z_ = np.random.normal(0, 1, (batch_size,1,1, 100))
                loss_g_, _ = sess.run([g_loss, g_optimizer], feed_dict={z: z_, x:x_, recon_weight:anneal_weight, isTrain:True})
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
f = open(root+'AccuracyResults.txt', 'w')
f.write('end of training-Train Accuracy Assuming all data labeled= '+ str(acc)+"\n")
f.write('end of training-Test Accuracy Assuming all data labeled= '+ str(test_acc)+"\n")
f.close()
print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']),
                                                                  training_epoch, total_ptime))
print("Training finish!... save training results")
with open(root +  'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)
show_train_hist(train_hist, save=True, path=root +  'train_hist.png')

images = []
for e in range(training_epoch):
    img_name = root + 'Fixed_results/'+  str(e) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + 'generation_animation.gif', images, fps=5)

sess.close()
