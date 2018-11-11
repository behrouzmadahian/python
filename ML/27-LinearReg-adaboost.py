import pandas as pd, numpy as np, random
import  time
import utilityFuncs1_noValidation as utilityFuncs1
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
# print(hiddenLayerSizes)
# dataset=pd.read_csv('GARCH_ret_series.csv')

##############
def LSTM_CNN_pred(dataset, lookback=30, lookahead=1,
                  trainSplit=0.6,  run=1):
    '''
    :param dataset: input: a n*r matrix with each column a different time series
    The first column is the time variable and second column is the coded time variable rom 0- end
    , used in determining the dates on train, validation, and test
    data.
    the last column is the close price!
    '''

    train, test = utilityFuncs1.MultivarSeries_to_tensor1(dataset, trainSplit=trainSplit,
                                                                      lookback=lookback,
                                                                     lookahead=lookahead)
    train_x_tensor, train_Y, train_dates = train
    test_x_tensor, test_Y, test_dates = test

    ##########################################################################################
    print('Shape of Final data for modeling:')
    print(train_x_tensor.shape)
    print(test_x_tensor.shape)
    #num_features = train_x_tensor.shape[2]
    ###############################################################
    ##########################MODELING:############################
    ###############################################################
    plt.hist(train_Y)
    plt.show()
    model=AdaBoostRegressor(linear_model.LinearRegression(),n_estimators=9,loss='square')

    model.fit(train_x_tensor,train_Y)
    trainPred = model.predict(train_x_tensor )
    estimator_errors= model.estimator_errors_
    print(trainPred[0:5])
    print(estimator_errors)
    # print(trainPred[0:10])
    train_sign = np.sign(train_Y)
    train_sign = (train_sign + 1) * train_sign / 2  # make them zero one!
    # print(train_sign[0:10])
    train_sign_pred = np.sign(trainPred)
    train_sign_pred = (train_sign_pred + 1) * train_sign_pred / 2
    # print(train_sign_pred[0:10])
    print(train_Y.shape)
    print(train_sign.shape, train_sign_pred.shape)
    trainRes = {'yTrain': train_Y[:, 0], 'yTrain_sign': train_sign[:, 0], 'yTrain_pred': trainPred,
                'yTrain_sign_pred': train_sign_pred, 'Dates:': train_dates}
    trainRes_df = pd.DataFrame(trainRes)
    trainRes_df.to_csv('Train_pred_CNNLSTM_3CAT_%d.csv' % run, index=False)

    testPred = model.predict(test_x_tensor)
    test_sign = np.sign(test_Y)
    test_sign = (test_sign + 1) * test_sign / 2
    test_sign_pred = np.sign(testPred)
    test_sign_pred = (test_sign_pred + 1) * test_sign_pred / 2

    testRes = {'yTest': test_Y[:, 0], 'yTest_sign': test_sign[:, 0], 'yTest_pred': testPred,
               'yTest_sign_pred': test_sign_pred, 'Dates': test_dates}
    testRes_df = pd.DataFrame(testRes)
    testRes_df.to_csv('Test_pred_CNNLSTM_3CAT_%d.csv' % run, index=False)
    # we need to transform the predictions into the original scale and calculate MSE:
    return trainRes_df,  testRes_df,\
           train_x_tensor.shape[0],test_x_tensor.shape[0]
    ##########


lookback = range(30, 31)
lookahead = range(1, 2)

start = time.time()
HowMany_runs = 1
dataset = pd.read_csv('SQ_Adj_forModeling_new.csv')
MSE_grid=np.ones(HowMany_runs)*100
for kk in range(HowMany_runs):
    print ('RUN:  ',kk)
    for i in lookback:

        for j in lookahead:
            print('-' * 40)
            print('Look_back: ', i)
            print('Look_ahead: ', j)
            print('-' * 40)
            # evaluates the model and writes the results to file!
            trainRes_df,  testRes_df,train_L,test_L = \
                LSTM_CNN_pred(dataset, lookback=i, lookahead=j,  run=kk,trainSplit=0.6)

            if kk==0:
                Train_Pred_sign_grid = np.zeros((train_L, 3 + HowMany_runs)) * (-100)
                Train_Pred_sign_grid[:, 0:3] = trainRes_df.values[:, (0,1,3)] #date y, y_sign
                Train_Pred_sign_grid[:, 3] = trainRes_df.values[:, -1] #y_pred_sign

                Train_Pred_grid = np.zeros((train_L, 3 + HowMany_runs)) * (-100)
                Train_Pred_grid[:, 0:3] = trainRes_df.values[:, (0, 1, 3)]  # date y, y_sign
                Train_Pred_grid[:, 3] = trainRes_df.values[:, 2]  # y_pred

                Test_Pred_sign_grid = np.zeros((test_L, 3 + HowMany_runs)) * (-100)
                Test_Pred_sign_grid[:, 0:3] = testRes_df.values[:, (0,1,3)]
                Test_Pred_sign_grid[:, 3] = testRes_df.values[:, -1] #y pred sign

                Test_Pred_grid = np.zeros((test_L, 3 + HowMany_runs)) * (-100)
                Test_Pred_grid[:, 0:3] = testRes_df.values[:, (0, 1, 3)]
                Test_Pred_grid[:, 3] = testRes_df.values[:, 2] #y pred
            else:
                Train_Pred_sign_grid[:, kk + 3] = trainRes_df.values[:, -1] #y pred sign
                Train_Pred_grid[:,kk + 3]=trainRes_df.values[:, 2] #y pred

                Test_Pred_sign_grid[:, kk + 3] = testRes_df.values[:, -1]
                Test_Pred_grid[:, kk + 3] = testRes_df.values[:, 2]




end = time.time()

print('ELAPSED time(Minutes): ', (end - start) / 60.)
print('Elapsed time per training (Minutes): ', ((end - start) / (60.0 *HowMany_runs)))

np.savetxt('TrainPred_Grid_sign_diff_seeds.txt',Train_Pred_sign_grid,delimiter='\t')
np.savetxt('TrainPred_Grid_diff_seeds.txt',Train_Pred_grid,delimiter='\t')

np.savetxt('TestPred_Grid_sign_diff_seeds.txt',Test_Pred_sign_grid,delimiter='\t')
np.savetxt('TestPred_Grid_diff_seeds.txt',Test_Pred_grid,delimiter='\t')

#np.savetxt('Train_MSE_grid_diff_seeds.txt',MSE_grid,delimiter='\t')

