import tensorflow as tf
v1=tf.Variable(tf.ones([100]),name='v1')
v2=tf.Variable(tf.zeros([100]),name='v2')
#add op to initialize:
init_op=tf.global_variables_initializer()
#add ops to save and restore all the variables:
saver=tf.train.Saver()
#later launch the model, initialize the variables and do some work, save the variable to disk:
with tf.Session() as sess:
    sess.run(init_op)
    #saving variables:
    save_path=saver.save(sess,'./model_checkPoint.ckpt')
    print(save_path)

#############################
#CHOOSING VARIABLES TO SAVE AND RESTORE:
#if you do not pass any argument to tf.train.Saver() the save handles all variables in the graph.
#each one of them is saved under the name that was passed when the variable was created.
#you may want to save a variable named 'weights'  and restore its value later into variable 'params'
#it is also sometimes useful to only save or restore a subset of variables used by the model.
#EXAMPLE:
#you may have trained a NN with 5 layers and you now want to train a new model with 6 layers restoring the parameters
#from the 5 layers of the previously trained model into the first 5 layers of the new model.
#pass  a dictionary to tf.train.Saver().
#keys are names to use and values are the variables to manage.
#NOTE:
#you can create as many saver objects as you want if you need to save and restore different subsets of the model variables
#the same variable can be listed in multiple saver objects, its value is only changed when the saver 'restore()'
#method is run.
#if you only restore a subset of the model_variables at the start of a session, you have to run an initialize op
#for the other variables.
#EXAMPLE:
v11=tf.Variable(tf.zeros([20]),name='v11')
v22=tf.Variable(tf.ones([20]),name='v22')
init_op=tf.global_variables_initializer()
saver=tf.train.Saver({'myV11':v11})
sess1=tf.Session()
sess1.run(init_op)
    #saving variables:
save_path=saver.save(sess1,'./Saving_specific_vars.ckpt')





