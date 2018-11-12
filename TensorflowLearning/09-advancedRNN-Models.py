import tensorflow as tf


def MLP(x, y, weights, biases, curr_optimizer, learning_rate, objective, batch_size, markets,
        activation=tf.nn.tanh, l2Reg = 0.01, l2RegOutput = 0., l2Reg_biases = 0. ):
    '''
    :param x: placeholder tensor for input
    :param y_for_train:  is 1-day return for optimizing for sharpe
    :param weights: dictionary of all the weight tensors in the model
    :param biases: dictionary of all the bias tensors in the model
    :param learning_rate:  placeholder for learning rate, we will feed this at training!
    :return: returns the output of the model (just the logits before the softmax (linear output)!
    '''
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = activation(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = activation(layer_2)

    output = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    #output = activation(output)

    sharpe_loss = objective(output, y, batch_size, len(markets))


    # l2 regularization loss:
    # I will regularize the output layer by 0.1 of l2Reg

    l2Loss = sum(tf.nn.l2_loss(curr_var)
                 for curr_var in tf.trainable_variables()
                 if not ('W_out' in curr_var.name or 'B_' in curr_var.name))

    l2Loss_output_layer = sum(tf.nn.l2_loss(curr_var)
                              for curr_var in tf.trainable_variables()
                              if ('W_out' in curr_var.name))

    l2Loss_Biases = sum(tf.nn.l2_loss(curr_var)
                        for curr_var in tf.trainable_variables()
                        if ('B_' in curr_var.name))

    l2Loss = l2Reg * l2Loss + l2RegOutput * l2Loss_output_layer + l2Reg_biases * l2Loss_Biases



    sharpe_plus_l2_loss = sharpe_loss + l2Loss

    # cost function and optimization

    optimizer = curr_optimizer(learning_rate=learning_rate).minimize(sharpe_plus_l2_loss)

    return optimizer, output, sharpe_plus_l2_loss


def vanilla_LSTM(x, y, weights, biases, keep_prob ,curr_optimizer, learning_rate, objective, batch_size, markets,
        activation=tf.nn.tanh, l2Reg = 0.01, l2RegOutput = 0., l2Reg_biases = 0., lookback = 30, hidden_size = 5,
                         n_layers = 2):

    # input is  batch_size*lookback -> need to convert to  (batch_size, lookback, features)
    #check to see if the data is already in appropriate shape:
    if len(x.get_shape()) == 3:
        print('X already 3 dimesnional')

    else:
        x = tf.reshape(x, [-1, lookback, 1])

    # need to convert input of shape (batch_size, time_steps,features)
    # to list of (batch_size, features) of length time_steps

    x = tf.unstack(x, lookback, axis = 1)

    def lstm_cell():

        cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, activation=activation)

        #if applying dropout after each layer is NOT desired, remove this and
        #maybe apply drop out to the output of the stack!

        cell = tf.nn.rnn_cell.DropoutWrapper(cell = cell, output_keep_prob = keep_prob)
        return cell

    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(n_layers)])
    lstm_out, states = tf.contrib.rnn.static_rnn(stacked_lstm, x,  dtype=tf.float32)

    print('Shape of output tensor:', len(lstm_out), lstm_out[0].get_shape())

    print('shape of state tensor: ( (h1, c1), (h2, c2),.. ) '
          '  the final state (c_i) of shape (batch_size, n_hidden)=',
          states[1][0].get_shape())

    #if no dropout applied at each layer and applying at final layer output is desired
    #lstm_out= tf.nn.dropout(lstm_out[-1], keep_prob = dropout)

    output = tf.add(tf.matmul(lstm_out[-1], weights['out']), biases['out'])

    sharpe_loss = objective(output, y, batch_size, len(markets))

    # l2 regularization loss:
    # I will regularize the output layer by 0.1 of l2Reg

    l2Loss = sum(tf.nn.l2_loss(curr_var)
                 for curr_var in tf.trainable_variables()
                 if not ('W_out' in curr_var.name or 'B_' in curr_var.name))

    l2Loss_output_layer = sum(tf.nn.l2_loss(curr_var)
                              for curr_var in tf.trainable_variables()
                              if ('W_out' in curr_var.name))

    l2Loss_Biases = sum(tf.nn.l2_loss(curr_var)
                        for curr_var in tf.trainable_variables()
                        if ('B_' in curr_var.name))

    l2Loss = l2Reg * l2Loss + l2RegOutput * l2Loss_output_layer + l2Reg_biases * l2Loss_Biases


    sharpe_plus_l2_loss = sharpe_loss + l2Loss

    # cost function and optimization

    optimizer = curr_optimizer(learning_rate=learning_rate).minimize(sharpe_plus_l2_loss)

    return optimizer, output, sharpe_plus_l2_loss


def vanilla_bidirectional_LSTM_1layer(x, y, weights, biases, keep_prob ,curr_optimizer, learning_rate, objective, batch_size, markets,
        activation=tf.nn.tanh, l2Reg = 0.01, l2RegOutput = 0., l2Reg_biases = 0., lookback = 30, hidden_size = 5):

    # input is  batch_size*lookback -> need to convert to  (batch_size, lookback, features)
    x = tf.reshape(x, [-1, lookback, 1])

    # need to convert input of shape (batch_size, time_steps,features)
    # to list of (batch_size, features) of length time_steps

    x = tf.unstack(x, lookback, axis = 1)

    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, activation=activation)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, activation=activation)
    lstm_out, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                             lstm_bw_cell,
                                                             x,
                                                             dtype=tf.float32,
                                                             initial_state_fw=None,
                                                             initial_state_bw=None)

    #if no dropout applied at each layer and applying at final layer output is desired
    #lstm_out= tf.nn.dropout(lstm_out[-1], keep_prob = dropout)

    output = tf.add(tf.matmul(lstm_out[-1], weights['out']), biases['out'])

    sharpe_loss = objective(output, y, batch_size, len(markets))

    # l2 regularization loss:
    # I will regularize the output layer by 0.1 of l2Reg

    l2Loss = sum(tf.nn.l2_loss(curr_var)
                 for curr_var in tf.trainable_variables()
                 if not ('W_out' in curr_var.name or 'B_' in curr_var.name))

    l2Loss_output_layer = sum(tf.nn.l2_loss(curr_var)
                              for curr_var in tf.trainable_variables()
                              if ('W_out' in curr_var.name))

    l2Loss_Biases = sum(tf.nn.l2_loss(curr_var)
                        for curr_var in tf.trainable_variables()
                        if ('B_' in curr_var.name))

    l2Loss = l2Reg * l2Loss + l2RegOutput * l2Loss_output_layer + l2Reg_biases * l2Loss_Biases


    sharpe_plus_l2_loss = sharpe_loss + l2Loss

    # cost function and optimization

    optimizer = curr_optimizer(learning_rate=learning_rate).minimize(sharpe_plus_l2_loss)

    return optimizer, output, sharpe_plus_l2_loss


def peephole_LSTM(x, y, weights, biases, keep_prob ,curr_optimizer, learning_rate, objective, batch_size, markets,
        activation=tf.nn.tanh, l2Reg = 0.01, l2RegOutput = 0., l2Reg_biases = 0., lookback = 30, hidden_size = 5,
                         n_layers = 2):

    # input is  batch_size*lookback -> need to convert to  (batch_size, lookback, features)
    x = tf.reshape(x, [-1, lookback, 1])

    # need to convert input of shape (batch_size, time_steps,features)
    # to list of (batch_size, features) of length time_steps

    x = tf.unstack(x, lookback, axis = 1)

    def lstm_cell():

        cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0, activation=activation, use_peepholes=True)

        #if applying dropout after each layer is NOT desired, remove this and
        #maybe apply drop out to the output of the stack!

        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob = keep_prob)
        return cell

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell() for _ in range(n_layers)])


    lstm_out, states = tf.contrib.rnn.static_rnn(stacked_lstm, x,  dtype=tf.float32)

    print('Shape of output tensor:', len(lstm_out), lstm_out[0].get_shape())

    print('shape of state tensor: ( (h1, c1), (h2, c2),.. ) '
          '  the final state (c_i) of shape (batch_size, n_hidden)=',
          states[1][0].get_shape())

    #if no dropout applied at each layer and applying at final layer output is desired
    #lstm_out= tf.nn.dropout(lstm_out[-1], keep_prob = dropout)

    output = tf.add(tf.matmul(lstm_out[-1], weights['out']), biases['out'])

    sharpe_loss = objective(output, y, batch_size, len(markets))

    # l2 regularization loss:
    # I will regularize the output layer by 0.1 of l2Reg

    l2Loss = sum(tf.nn.l2_loss(curr_var)
                 for curr_var in tf.trainable_variables()
                 if not ('W_out' in curr_var.name or 'B_' in curr_var.name))

    l2Loss_output_layer = sum(tf.nn.l2_loss(curr_var)
                              for curr_var in tf.trainable_variables()
                              if ('W_out' in curr_var.name))

    l2Loss_Biases = sum(tf.nn.l2_loss(curr_var)
                        for curr_var in tf.trainable_variables()
                        if ('B_' in curr_var.name))

    l2Loss = l2Reg * l2Loss + l2RegOutput * l2Loss_output_layer + l2Reg_biases * l2Loss_Biases


    sharpe_plus_l2_loss = sharpe_loss + l2Loss

    # cost function and optimization

    optimizer = curr_optimizer(learning_rate=learning_rate).minimize(sharpe_plus_l2_loss)

    return optimizer, output, sharpe_plus_l2_loss


def peephole_bidirectional_LSTM_1layer(x, y, weights, biases, keep_prob, curr_optimizer, learning_rate, objective,
                                       batch_size, markets,
                                       activation=tf.nn.tanh, l2Reg=0.01, l2RegOutput=0., l2Reg_biases=0., lookback=30,
                                       hidden_size=5):
    # input is  batch_size*lookback -> need to convert to  (batch_size, lookback, features)
    x = tf.reshape(x, [-1, lookback, 1])

    # need to convert input of shape (batch_size, time_steps,features)
    # to list of (batch_size, features) of length time_steps

    x = tf.unstack(x, lookback, axis=1)

    lstm_fw_cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0, activation=activation, use_peepholes=True)
    lstm_bw_cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0, activation=activation, use_peepholes=True)

    lstm_out, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                             lstm_bw_cell,
                                                             x,
                                                             dtype=tf.float32,
                                                             initial_state_fw=None,
                                                             initial_state_bw=None)

    # if no dropout applied at each layer and applying at final layer output is desired
    # lstm_out= tf.nn.dropout(lstm_out[-1], keep_prob = dropout)

    output = tf.add(tf.matmul(lstm_out[-1], weights['out']), biases['out'])

    sharpe_loss = objective(output, y, batch_size, len(markets))

    # l2 regularization loss:
    # I will regularize the output layer by 0.1 of l2Reg

    l2Loss = sum(tf.nn.l2_loss(curr_var)
                 for curr_var in tf.trainable_variables()
                 if not ('W_out' in curr_var.name or 'B_' in curr_var.name))

    l2Loss_output_layer = sum(tf.nn.l2_loss(curr_var)
                              for curr_var in tf.trainable_variables()
                              if ('W_out' in curr_var.name))

    l2Loss_Biases = sum(tf.nn.l2_loss(curr_var)
                        for curr_var in tf.trainable_variables()
                        if ('B_' in curr_var.name))

    l2Loss = l2Reg * l2Loss + l2RegOutput * l2Loss_output_layer + l2Reg_biases * l2Loss_Biases

    sharpe_plus_l2_loss = sharpe_loss + l2Loss

    # cost function and optimization

    optimizer = curr_optimizer(learning_rate=learning_rate).minimize(sharpe_plus_l2_loss)

    return optimizer, output, sharpe_plus_l2_loss


def attention_LSTM(x, y, weights, biases, keep_prob ,curr_optimizer, learning_rate, objective, batch_size, markets,
        activation=tf.nn.tanh, l2Reg = 0.01, l2RegOutput = 0., l2Reg_biases = 0., lookback = 30, hidden_size = 5,
                         n_layers = 2, attention_length = 30):

    # input is  batch_size*lookback -> need to convert to  (batch_size, lookback, features)
    x = tf.reshape(x, [-1, lookback, 1])

    # need to convert input of shape (batch_size, time_steps,features)
    # to list of (batch_size, features) of length time_steps

    x = tf.unstack(x, lookback, axis=1)

    def attention_lstm_cell():
        cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0, activation=activation)
        cell = tf.contrib.rnn.AttentionCellWrapper(cell, attn_length=attention_length)

        # if applying dropout after each layer is NOT desired, remove this and
        # maybe apply drop out to the output of the stack!

        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=keep_prob)
        return cell

    stacked_lstm = tf.contrib.rnn.MultiRNNCell([attention_lstm_cell() for _ in range(n_layers)])
    lstm_out, states = tf.contrib.rnn.static_rnn(stacked_lstm, x,  dtype=tf.float32)

    print('Shape of output tensor:', len(lstm_out), lstm_out[0].get_shape())

    # print('shape of state tensor: ( (h1, c1), (h2, c2),.. ) '
    #       '  the final state (c_i) of shape (batch_size, n_hidden)=',
    #       states[1][0].get_shape())

    #if no dropout applied at each layer and applying at final layer output is desired

    #lstm_out= tf.nn.dropout(lstm_out[-1], keep_prob = dropout)

    output = tf.add(tf.matmul(lstm_out[-1], weights['out']), biases['out'])

    sharpe_loss = objective(output, y, batch_size, len(markets))

    # l2 regularization loss:
    # I will regularize the output layer by 0.1 of l2Reg

    l2Loss = sum(tf.nn.l2_loss(curr_var)
                 for curr_var in tf.trainable_variables()
                 if not ('W_out' in curr_var.name or 'B_' in curr_var.name))

    l2Loss_output_layer = sum(tf.nn.l2_loss(curr_var)
                              for curr_var in tf.trainable_variables()
                              if ('W_out' in curr_var.name))

    l2Loss_Biases = sum(tf.nn.l2_loss(curr_var)
                        for curr_var in tf.trainable_variables()
                        if ('B_' in curr_var.name))

    l2Loss = l2Reg * l2Loss + l2RegOutput * l2Loss_output_layer + l2Reg_biases * l2Loss_Biases


    sharpe_plus_l2_loss = sharpe_loss + l2Loss

    # cost function and optimization

    optimizer = curr_optimizer(learning_rate=learning_rate).minimize(sharpe_plus_l2_loss)

    return optimizer, output, sharpe_plus_l2_loss


def bidirectional_attention_LSTM(x, y, weights, biases, keep_prob ,curr_optimizer, learning_rate, objective, batch_size, markets,
        activation=tf.nn.tanh, l2Reg = 0.01, l2RegOutput = 0., l2Reg_biases = 0., lookback = 30, hidden_size = 5,
                         attention_length = 30):

    # input is  batch_size*lookback -> need to convert to  (batch_size, lookback, features)
    x = tf.reshape(x, [-1, lookback, 1])

    # need to convert input of shape (batch_size, time_steps,features)
    # to list of (batch_size, features) of length time_steps

    x = tf.unstack(x, lookback, axis = 1)

    lstm_fw_cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0, activation=activation, use_peepholes=True)
    lstm_fw_cell = tf.contrib.rnn.AttentionCellWrapper(lstm_fw_cell, attn_length=attention_length)
    lstm_bw_cell = tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=1.0, activation=activation, use_peepholes=True)
    lstm_bw_cell = tf.contrib.rnn.AttentionCellWrapper(lstm_bw_cell, attn_length=attention_length)

    lstm_out, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell,
                                                             lstm_bw_cell,
                                                             x,
                                                             dtype=tf.float32,
                                                             initial_state_fw=None,
                                                             initial_state_bw=None)
    #if no dropout applied at each layer and applying at final layer output is desired

    #lstm_out= tf.nn.dropout(lstm_out[-1], keep_prob = dropout)

    output = tf.add(tf.matmul(lstm_out[-1], weights['out']), biases['out'])

    sharpe_loss = objective(output, y, batch_size, len(markets))

    # l2 regularization loss:
    # I will regularize the output layer by 0.1 of l2Reg

    l2Loss = sum(tf.nn.l2_loss(curr_var)
                 for curr_var in tf.trainable_variables()
                 if not ('W_out' in curr_var.name or 'B_' in curr_var.name))

    l2Loss_output_layer = sum(tf.nn.l2_loss(curr_var)
                              for curr_var in tf.trainable_variables()
                              if ('W_out' in curr_var.name))

    l2Loss_Biases = sum(tf.nn.l2_loss(curr_var)
                        for curr_var in tf.trainable_variables()
                        if ('B_' in curr_var.name))

    l2Loss = l2Reg * l2Loss + l2RegOutput * l2Loss_output_layer + l2Reg_biases * l2Loss_Biases


    sharpe_plus_l2_loss = sharpe_loss + l2Loss

    # cost function and optimization

    optimizer = curr_optimizer(learning_rate=learning_rate).minimize(sharpe_plus_l2_loss)

    return optimizer, output, sharpe_plus_l2_loss



