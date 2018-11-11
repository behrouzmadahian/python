import numpy as np
import pandas as pd
import dataProcessing
import copy

class Window_train_data(object):
    '''
    I assume, all markets start from the same date and thus date column is EXACtly the SAME!
    '''
    def __init__ (self, lookback = 30, look_ahead = 1, sd_window = 100,train_window = 500, slide_window = 100,
                  markets = ('SQ', 'NQ', 'MQ', 'DQ', 'RN'), nRandom_start = 25):


        self.data = {}

        self.pred_start = [] # keeps the start date of test window <slide window>
        self.train_window = train_window
        self.slide_window = slide_window
        self.train_start_ind = 0 # index of start of training
        self.markets = markets
        self.nPreds = nRandom_start
        self.test_end_ind = 0
        self.test_start_ind = 0  # index where test period starts
        self.lookback = lookback
        self.lookahead = look_ahead
        self.pre_processing_sd_window = sd_window

        try:
            dates1 = pd.read_csv(
                'C:/behrouz/Projects/DailyModels_new/NeuralNet/tf-5Market-model-epoch-CP-onlineTraining/CommonDates_DQ.csv')

            dates2 = pd.read_csv(
                'C:/behrouz/Projects/DailyModels_new/NeuralNet/tf-5Market-model-epoch-CP-onlineTraining/CommonDates_SQ.csv')

        except IOError:
            print('Can not Open  the dates files to match markets..')

        for i in range(len(markets)):

            # Some markets are missing some dates that screqs everythning UP!!!
            # I get the date for DQ that has minimum data for the SAME dates!!!

            try:
                datadir = 'C:/behrouz/Projects/DailyModels_new/NeuralNet/tf-5Market-model-epoch-CP-onlineTraining/%s-all-Adj.csv'

                curr_data = pd.read_csv(datadir % markets[i])

            except  IOError:

                print('Can not Open  the datafile for market= %s', markets[i])

            curr_data =pd.merge(curr_data, dates1, on = 'dtStart', how = 'inner')
            curr_data =pd.merge(curr_data, dates2, on = 'dtStart', how = 'inner')

            # returns dataX, dataY, ret1_day, Pred_start_dates
            curr_market_data = dataProcessing.time_series_toMatrix_noTest(curr_data,
                                                                          self.lookback,
                                                                          self.lookahead,
                                                                          self.pre_processing_sd_window)
            self.data[self.markets[i]] = curr_market_data

        # make sure length of data is divisible by slide_window!
        # for simplicity
        # or I will add the last slide that is less than
        # the window to the last complete window

        self.data_end_ind = self.data[self.markets[0]][0].shape[0]
        self.dates = self.data[self.markets[0]][3]

    def next_train_window(self):

        ''' returns the data of size train_slide window,
            if it is not the first slide,
            repeats the last test window and incorporate into the train window.
            keeps track of test start date
        '''
        test_data_dict = {}
        train_data_dict = {}

        if self.train_start_ind == 0:
            self.test_start_ind = self.train_window + self.train_start_ind
            self.test_end_ind =self.test_start_ind + self.slide_window

            # print('Data End, Train Start, Train End, Test END')
            #
            # print(self.data_end_ind, self.train_start_ind, self.test_start_ind, self.test_end_ind)

            trainX = copy.deepcopy(self.data[self.markets[0]][0][self.train_start_ind: self.test_start_ind])
            trainY = copy.deepcopy(self.data[self.markets[0]][1][self.train_start_ind: self.test_start_ind])
            train_ret_1day = copy.deepcopy(self.data[self.markets[0]][2][self.train_start_ind: self.test_start_ind])
            train_dates = copy.deepcopy(self.dates [ self.train_start_ind: self.test_start_ind ])

            testX = copy.deepcopy(self.data[self.markets[0]][0][self.test_start_ind : self.test_end_ind])
            testY = copy.deepcopy(self.data[self.markets[0]][1][self.test_start_ind : self.test_end_ind])
            test_ret_1day = copy.deepcopy(self.data[self.markets[0]][2][self.test_start_ind : self.test_end_ind])
            test_dates = copy.deepcopy(self.dates [ self.test_start_ind : self.test_end_ind])

            test_data_dict[self.markets[0] ] = (testX, testY, test_ret_1day, test_dates)

            train_data_dict[self.markets[0] ] = (trainX, trainY, train_ret_1day, train_dates)

            for i in range(1, len(self.markets), 1):

                curr_trainX = copy.deepcopy(self.data[self.markets[i]][0][self.train_start_ind: self.test_start_ind])
                curr_trainY = copy.deepcopy(self.data[self.markets[i]][1][self.train_start_ind: self.test_start_ind])
                curr_train_ret_1day = copy.deepcopy(self.data[self.markets[i]][2][self.train_start_ind: self.test_start_ind])

                curr_testX = copy.deepcopy(self.data[self.markets[i]][0][self.test_start_ind : self.test_end_ind])
                curr_testY = copy.deepcopy(self.data[self.markets[i]][1][self.test_start_ind : self.test_end_ind])

                curr_test_ret_1day = copy.deepcopy(self.data[self.markets[i]][2][self.test_start_ind : self.test_end_ind])

                trainX = np.append(trainX, curr_trainX, axis = 0)
                trainY = np.append(trainY, curr_trainY, axis = 0)
                train_ret_1day = np.append(train_ret_1day, curr_train_ret_1day, axis = 0)

                # Current market data:

                test_data_dict[self.markets[i]] = (curr_testX, curr_testY, curr_test_ret_1day, test_dates)
                train_data_dict[self.markets[i]] = (curr_trainX, curr_trainY, curr_train_ret_1day, train_dates)

            self.pred_start.append(self.dates[self.test_start_ind])
            self.train_start_ind += self.slide_window


            return trainX, trainY, train_ret_1day, train_dates, train_data_dict,test_data_dict

        else:

            if self.data_end_ind - self.train_start_ind > self.train_window:

                self.test_start_ind = self.train_start_ind + self.train_window

                train_dates = copy.deepcopy(self.dates[self.train_start_ind: self.test_start_ind])

                trainX = copy.deepcopy(self.data[self.markets[0]][0][self.train_start_ind: self.test_start_ind])
                trainY = copy.deepcopy(self.data[self.markets[0]][1][self.train_start_ind : self.test_start_ind])
                train_ret_1day = copy.deepcopy(self.data[self.markets[0]][2][self.train_start_ind : self.test_start_ind])

                #add a duplicate of newly added data so that the model sees more of it!

                duplicate_pieceX = copy.deepcopy(self.data[self.markets[0]][0][
                                                 self.test_start_ind - self.slide_window : self.test_start_ind])

                trainX = np.append(trainX, duplicate_pieceX, axis = 0)

                duplicate_pieceY = copy.deepcopy(self.data[self.markets[0]][1][
                                                 self.test_start_ind - self.slide_window : self.test_start_ind])

                trainY = np.append(trainY, duplicate_pieceY, axis = 0)

                duplicate_piece_ret_1day = copy.deepcopy(self.data[self.markets[0]][2][
                                                         self.test_start_ind - self.slide_window : self.test_start_ind])

                train_ret_1day = np.append(train_ret_1day, duplicate_piece_ret_1day, axis = 0)

                train_data_dict[self.markets[0]] = (trainX, trainY, train_ret_1day, train_dates)


                if self.data_end_ind - self.test_start_ind <= self.slide_window:

                    self.test_end_ind = self.data[self.markets[0]][0].shape[0]
                    # print('Error is HERE!')
                    # print ('Data End, Train Start, Test Start, Test END')
                    #
                    # print(self.data_end_ind, self.train_start_ind,  self.test_start_ind, self.test_end_ind)

                    test_dates = copy.deepcopy(self.dates[self.test_start_ind: self.test_end_ind])
                    testX =copy.deepcopy( self.data[self.markets[0]][0][self.test_start_ind: self.test_end_ind])
                    testY = copy.deepcopy(self.data[self.markets[0]][1][self.test_start_ind: self.test_end_ind])
                    test_ret_1day = copy.deepcopy(self.data[self.markets[0]][2][self.test_start_ind: self.test_end_ind])

                else:
                    self.test_end_ind = self.test_start_ind + self.slide_window
                    # print ('Data End, Train Start, Test Start, Test END')
                    # print(self.data_end_ind, self.train_start_ind,  self.test_start_ind, self.test_end_ind)


                    test_dates = copy.deepcopy(self.dates[self.test_start_ind: self.test_end_ind])
                    testX = copy.deepcopy(self.data[self.markets[0]][0][self.test_start_ind: self.test_end_ind])
                    testY = copy.deepcopy(self.data[self.markets[0]][1][self.test_start_ind: self.test_end_ind])
                    test_ret_1day = copy.deepcopy(self.data[self.markets[0]][2][self.test_start_ind: self.test_end_ind])

                test_data_dict [self.markets[0]] = (testX, testY, test_ret_1day, test_dates)

                for i in range(1, len(self.markets), 1):

                    curr_trainX = copy.deepcopy(self.data[self.markets[i]][0][self.train_start_ind: self.test_start_ind])
                    curr_trainY = copy.deepcopy(self.data[self.markets[i]][1][self.train_start_ind: self.test_start_ind])
                    curr_train_ret_1day = copy.deepcopy(self.data[self.markets[i]][2][self.train_start_ind: self.test_start_ind])

                    curr_duplicate_pieceX = copy.deepcopy(self.data[self.markets[i]][0][
                                                          self.test_start_ind - self.slide_window : self.test_start_ind])

                    curr_trainX = np.append(curr_trainX, curr_duplicate_pieceX, axis = 0)


                    curr_duplicate_pieceY = copy.deepcopy(self.data[self.markets[i]][1][
                                                          self.test_start_ind - self.slide_window : self.test_start_ind])

                    curr_trainY = np.append(curr_trainY, curr_duplicate_pieceY, axis = 0)


                    curr_duplicate_piece_ret_1day = copy.deepcopy(self.data[self.markets[i]][2][
                                                                  self.test_start_ind - self.slide_window : self.test_start_ind])

                    curr_train_ret_1day = np.append(curr_train_ret_1day, curr_duplicate_piece_ret_1day, axis = 0)

                    trainX = np.append(trainX, curr_trainX, axis = 0)
                    trainY = np.append(trainY, curr_trainY, axis = 0)
                    train_ret_1day = np.append(train_ret_1day, curr_train_ret_1day, axis = 0)

                    train_data_dict[self.markets[i]] = (curr_trainX, curr_trainY, curr_train_ret_1day, train_dates)

                    if self.data_end_ind - self.test_start_ind <= self.slide_window:


                        curr_testX = copy.deepcopy(self.data[self.markets[i]][0][self.test_start_ind: ])
                        curr_testY = copy.deepcopy(self.data[self.markets[i]][1][self.test_start_ind: ])
                        curr_test_ret_1day = copy.deepcopy(self.data[self.markets[i]][2][self.test_start_ind: ])

                    else:

                        curr_testX = copy.deepcopy(self.data[self.markets[i]][0][
                                                   self.test_start_ind: self.test_start_ind + self.slide_window])

                        curr_testY = copy.deepcopy(self.data[self.markets[i]][1][
                                                   self.test_start_ind: self.test_start_ind + self.slide_window])

                        curr_test_ret_1day = copy.deepcopy(self.data[self.markets[i]][2][
                                                           self.test_start_ind: self.test_start_ind + self.slide_window])

                    test_data_dict[self.markets[i]] = (curr_testX, curr_testY, curr_test_ret_1day, test_dates)

                self.pred_start.append(self.dates[self.test_start_ind])
                self.train_start_ind += self.slide_window

                return trainX, trainY, train_ret_1day, train_dates, train_data_dict,  test_data_dict

            else:
                return None

    def prediction_containers(self):
        pred_containers_dict = {}
        test_length = self.data_end_ind - self.train_window

        for i in range (len(self.markets)):
            tmp = np.zeros((test_length, self.nPreds + 2))
            tmp[:, 0] = self.data[self.markets[i]][3][self.train_window:]  # prediction dates of the market
            tmp[:, 1] = self.data[self.markets[i]][2][self.train_window:]  # 1 day returns

            pred_containers_dict[self.markets[i]] = tmp

        return pred_containers_dict

    def shuffle_train_dict(self, train_dict):

        ''' gets the training dictionary data and shuffles each markets by the same shuffled indicies
            returns a dictionary with following elements for each market: (trainX, trainY, ret_1day)
        '''

        a = np.arange(train_dict[self.markets[0]][0].shape[0])
        np.random.shuffle(a)
        curr_dict = {}
        for i in range(len(self.markets)):

            curr_dict[self.markets[i]] = (train_dict[self.markets[i]][0][a, :],
                                     train_dict[self.markets[i]][1][a],
                                     train_dict[self.markets[i]][2][a]
                                     )

        return curr_dict

    def next_batch_dict(self, train_dict, batch_number, batch_size, indY):

        '''
           indY: 1 for scaled return, 2 for 1-day return
           returns Xbatch and YBatch.
        '''

        total_batches = self.train_window // batch_size
        rem = self.train_window % batch_size

        if (batch_number + 1) == total_batches and rem != 0:

            xBatch = train_dict[self.markets[0]][0][(total_batches - 1) * batch_size + rem:, :]
            trainY_batch = train_dict[self.markets[0]][indY][(total_batches - 1) * batch_size + rem:]

            for i in range(1, len(self.markets), 1):

                xb = train_dict[self.markets[i]][0][(total_batches - 1) * batch_size + rem:, :]
                yb = train_dict[self.markets[i]][indY][(total_batches - 1) * batch_size + rem:]

                xBatch = np.append(xBatch, xb, axis=0)
                trainY_batch = np.append(trainY_batch, yb, axis=0)

        else:

            xBatch = train_dict[self.markets[0]][0][batch_number * batch_size: (batch_number + 1) * batch_size, :]
            trainY_batch = train_dict[self.markets[0]][indY][batch_number * batch_size: (batch_number + 1) * batch_size]

            for i in range(1, len(self.markets), 1):

                xb = train_dict[self.markets[i]][0][batch_number * batch_size: (batch_number + 1) * batch_size, :]
                yb = train_dict[self.markets[i]][indY][batch_number * batch_size: (batch_number + 1) * batch_size]

                xBatch = np.append(xBatch, xb, axis=0)
                trainY_batch = np.append(trainY_batch, yb, axis=0)

        return xBatch, trainY_batch



    def start_over(self):
         '''
         resets all the indicies to the begining of data
         '''
         self.train_start_ind = 0





if __name__ == '__main__':
   pass