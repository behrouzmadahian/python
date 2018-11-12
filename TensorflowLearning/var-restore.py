import tensorflow as tf
#RESTORING VARIABLES:#\
#the same saver object is used to restore variables.
#note that when you restore variables from a file, you do not have to initialize them beforehand.
#create variables:

# #later launch the model use the saver to restore variables from disk and do some work with the model:
with tf.Session() as sess:
    v1 = tf.Variable(tf.zeros([100]), name='v1')
    v2 = tf.Variable(tf.ones([100]), name='v2')
    saver = tf.train.Saver()

    #we need to initialize before being able to restore weights from file!
    tf.global_variables_initializer()
    saver.restore(sess,'./model_checkPoint.ckpt')
    print ('model Restored..')
    print ('doing some work on the model..')
    print(sess.run(v1))
    print(sess.run(v2))

#restoring specific variable saved with name myV11


with tf.Session() as sess1:
    v = tf.Variable(tf.ones([20]), name='myV11')
    saver1 = tf.train.Saver()
    tf.global_variables_initializer()
    saver1.restore(sess1,'./Saving_specific_vars.ckpt')
    print(sess1.run(v))