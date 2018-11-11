import numpy as np
import tensorflow as tf
import os, time, itertools, imageio, pickle
from matplotlib import  pyplot as plt
from tensorflow.examples.tutorials.mnist import  input_data
'''
paper: Improved techniques for training GANS!
the same as version 2 except:
-Assume Only the first 1000 samples in training data is labeled and the rest in unlabeled.
I use feature matching as the loss for Generator!
- Generator loss is feature matching loss
# we want to train a classifier than can train with few amount of labeled(supervised) data!
Here the following stipulations have been made:
1- Half of real data  has label and the other half is unlabled!.
2- the regular GAN loss the feature matching loss for Generator
##############
3- one-sided label smoothing : make sure this happens properly.
for the real unlabeld data, the loss is calculated as:
In the paper, for the real and unlabeld data they use 0.9 instead of 1 as label in calculating 
-0.9*log(P(Real)) = - -0.9 * log(1-P(fake)); p(fake)= p(y=k+1)
##############
Discriminator Loss:
d_loss = supervised Loss + unsupervised Loss
supervised_loss = -label* log(D(x)) < label is one-hot>
unsupervised loss = -log(1-D(x)|Y=k+1) <P(unlabled data is real>  + -log(D(G(Z)|Y =k+1)
Generator loss:
g_loss = abs(f_data-f_fake) from one of the middle layer activations

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
        '''
        nstead of directly applying a fully connected layer on top of the convolutions, we perform a
         Global Average Pooling (GAP) operation. Global Average Pooling is a regularization technique
          that has been used with success in some convolutional classifier nets as a replacement to 
          fully connected layers. In GAP we take the average over the spatial dimensions of a feature 
          map resulting in one value.
          In global average pooling, for every feature map we take the average over all the spatial
          domain and return a single value
          In: [BATCH_SIZE,HEIGHT X WIDTH X CHANNELS] --> [BATCH_SIZE, CHANNELS]
        '''
        GAP = tf.reduce_mean(lrelu3, axis = [1,2]) #global average pooling
        # Fully connected output layer
        class_logits = tf.layers.dense(GAP, n_classes, activation=None) # probability of each of 10 real classes!
        # sum of these is the  P(real) =1- p(fake)
        GAN_logits = tf.reduce_logsumexp(class_logits, axis =1)
        class_probs = tf.nn.softmax(class_logits)
        print('Discriminator: Shape of output of each convolution layers:')
        print(x.get_shape(),conv1.get_shape(), conv2.get_shape(), conv3.get_shape(),GAP.get_shape(),class_logits.get_shape())#, out.get_shape)
        return class_probs, class_logits, GAN_logits, GAP

fixed_Z = np.random.normal(0, 1, (25, 1, 1, 100))
x = tf.placeholder(tf.float32, shape = (None, 28,28,1))
label = tf.placeholder(tf.float32, shape=(None, n_classes))
z = tf.placeholder(tf.float32, shape = (None, 1, 1, 100))
label_mask = tf.placeholder(tf.int32, shape=(None))
isTrain = tf.placeholder(dtype=tf.bool)
#networks: generator:
G_z = generator(z, isTrain=isTrain)
fake_image = G_z
print('Shape of output of Generator:', G_z.get_shape())

# discriminator network:
d_real, d_real_logits, GAN_logits_real, GAP_feats_real  = discriminator(x, isTrain)
d_fake, d_fake_logits, GAN_logits_fake, GAP_feats_fake = discriminator(G_z, isTrain, reuse =True)
##
# Bulding the loss functions:
'''
we still need a way to represent the probability of an input image being real< For real unlabeled data>
and for the Generator rather <fake>, 
that is, we still need to model the binary classification problem for a regular GAN model.
'''
def build_loss(d_real, d_real_logits,GAN_logits_real,GAP_feats_real,
                GAN_logits_fake, GAP_feats_fake,
               label, label_mask, smooth = 0.9):
    # d_real: softmax output [B, n_class=10], d_real_logits: [B, n_class] # only real classes here!

    # Discriminator/ classifier loss:
    # 1. The loss for the GAN problem, where we minimize the cross-entropy for the binary
    #    real-vs-fake classification problem.
    #2. The loss for the digit classification problem, where we minimize the cross-entropy
    #     for the multi-class softmax.
    # onesided label smoothing on positive examples!
    real_unlabeled_loss  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=GAN_logits_real,
                                                                                 labels= tf.ones_like(GAN_logits_real)*smooth))
    fake_data_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=GAN_logits_fake,
                                                                            labels= tf.zeros_like(GAN_logits_fake)))
    unsupervised_loss = real_unlabeled_loss+ fake_data_loss
    #2- SupervisedLoss!
    supervised_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_real_logits, labels= label)
    label_mask = tf.squeeze(tf.to_float(label_mask)) # removes the dimensions of size 1!
    # print('label mask shape=', label_mask.get_shape())
    #
    supervised_loss = tf.reduce_sum(tf.multiply(supervised_loss, label_mask))

    # get the mean for labeled samples ONLY!!!
    supervised_loss = supervised_loss/ tf.maximum(1.0, tf.reduce_sum(label_mask))
    supervised_loss =tf.reshape(supervised_loss, [])

    d_loss = tf.reduce_sum(unsupervised_loss+ supervised_loss)
    # generator loss: Here original GAN loss!
    data_moments = tf.reduce_mean(GAP_feats_real, axis=0)
    sample_moments = tf.reduce_mean(GAP_feats_fake, axis=0)
    g_loss = tf.reduce_mean(tf.abs(data_moments - sample_moments))
    # classification accuracy:
    correct_prediction = tf.equal(tf.argmax(d_real, axis=1), tf.argmax(label, axis=1))
    accuracy_all_labeled = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # assuming all data is labeled!
    correct_masked = tf.reduce_sum(tf.cast(correct_prediction, tf.float32) * label_mask) # assuming both labeled and unlabeled!
    accuracy_correct_masked = correct_masked/tf.maximum(1.0, tf.reduce_sum(label_mask))

    return d_loss, g_loss, accuracy_all_labeled, accuracy_correct_masked


d_loss, g_loss, accuracy_all_labeled, accuracy_correct_masked  =build_loss(d_real, d_real_logits,GAN_logits_real,
                                                                           GAP_feats_real,
                                                                           GAN_logits_fake, GAP_feats_fake,
                                                                           label, label_mask, smooth = 0.9)
print(d_loss.get_shape(),'===')
# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

global_step = tf.train.get_or_create_global_step(graph=None)
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
root = 'C:/behrouz/Research_and_Development/GANS/codes/SSGAN/results_ver1_1/'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.exists(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')
if not os.path.exists(root+ 'Random_results'):
    os.mkdir(root+'Random_results')
if not os.path.exists(root+'checkpoint'):
    os.makedirs(root+'checkpoint')
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
saver_g = tf.train.Saver(G_vars)
saver_d = tf.train.Saver(D_vars)
saver_all = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    # MNIST resize and normalization
    #train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval()
    train_set, trainY = mnist.train.images, mnist.train.labels
    train_set = (train_set - 0.5) / 0.5
    label_mask_train = np.zeros(len(trainY))
    label_mask_train[:100] =1
    test_set, testY = mnist.test.images, mnist.test.labels
    test_set = (test_set - 0.5) / 0.5
    label_mask_test = np.ones(len(testY))
    np.random.seed(int(time.time()))
    start_time = time.time()
    inds = np.arange(len(train_set))
    for epoch in range(training_epoch):
        np.random.shuffle(inds)
        train_set = train_set[inds]
        trainY = trainY[inds]
        label_mask_train = label_mask_train[inds]
        z_ = sess.run(tf.random_uniform([batch_size, 1, 1, 100], minval=-1, maxval=1, dtype=tf.float32))
        train_acc, train_acc_masked, train_d_l, train_g_l = sess.run([accuracy_all_labeled, accuracy_correct_masked,d_loss, g_loss],
                                                   feed_dict={x:train_set,label:trainY, z:z_,
                                                        label_mask: label_mask_train,isTrain:False})
        test_acc, test_acc_masked,test_d_l, test_g_l = sess.run([accuracy_all_labeled, accuracy_correct_masked, d_loss, g_loss],
                                                feed_dict={x:test_set,label:testY, z:z_,
                                                             label_mask: label_mask_test, isTrain:False})
        # these are the results if assuming all data is labeled!
        print('Train Accuracy-all labeled=', train_acc,'Accuracy partial label=', train_acc_masked,
              'Train d loss= ', train_d_l, 'Train g loss= ', train_g_l)
        print('Test Accuracy=', test_acc, 'Accuracy partial label=', test_acc_masked,
              'Test d loss= ', test_d_l, 'Test g loss= ', test_g_l)
        print('-'*100)
        p = root + 'random_Results/' +  str(epoch) + '.png'
        fixed_p = root + 'Fixed_results/' +  str(epoch) + '.png'
        show_result(sess, (epoch + 1), show=False, path=p, isFix=False)
        show_result(sess, (epoch + 1), show=False, path=fixed_p, isFix=True)
        G_losses = []
        D_losses = []
        epoch_start_time = time.time()
        # update discriminator 3 times more often!

        for iter in range(train_set.shape[0]//batch_size):
            # update discriminator:
            x_ = train_set[iter*batch_size:(iter+1)*batch_size]
            y_ = trainY [iter*batch_size:(iter+1)*batch_size]
            l_mask_ = label_mask_train[iter*batch_size:(iter+1)*batch_size]
            z_ = sess.run(tf.random_uniform([batch_size,1,1, 100], minval = -1, maxval =1, dtype = tf.float32))

            loss_d_, _ = sess.run([d_loss, d_optimizer], feed_dict={x:x_,label:y_, z:z_,label_mask:l_mask_, isTrain:True})
            D_losses.append(loss_d_)
            # update generator
            # z_ = np.random.normal(0, 1, (batch_size,1,1, 100))
            loss_g_, _ = sess.run([g_loss, g_optimizer], feed_dict={z: z_,x:x_, isTrain:True})
            G_losses.append(loss_g_)

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        print('[%d/%d] - ptime: %.2f AVERAGE loss_d: %.3f, AVERAGE loss_g: %.3f' % ((epoch + 1), training_epoch, per_epoch_ptime,
                                                                    np.mean(D_losses), np.mean(G_losses)))

        train_hist['D_losses'].append(np.mean(D_losses))
        train_hist['G_losses'].append(np.mean(G_losses))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    saver_g.save(sess,root+'/check_point/generator.ckpt')
    saver_d.save(sess,root+'/check_point/discriminator.ckpt')
    saver_all.save(sess,root+'/check_point/generatorPlus_discriminator.ckpt')
f = open(root+'AccuracyResults.txt', 'w')
f.write('end of training -Train Accuracy on labeled part= '+ str(train_acc_masked)+"\n")
f.write('end of training-Train Accuracy Assuming all data labeled= '+ str(train_acc)+"\n")
f.write('end of training-Test Accuracy Assuming all data labeled= '+ str(test_acc)+"\n")
f.close()
end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

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
