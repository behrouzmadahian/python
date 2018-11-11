import tensorflow as tf

'''
Idea:
The idea is that any positive outcome with a prediction less than the 
maximum prediction of all the negative outcomes is contributing to a loss in the AUROC. 
'''


def rank_loss(y_true, y_pred_prob):
    # the maximum score for negative class outcome is
    max_prob_neg_class = tf.reduce_max(y_pred_prob * (y_true == 0))
    # translate into the raw scores before the logit
    y_pred_score = tf.log(y_pred_prob / (1 - y_pred_prob))

    # determine how much each score is above or below it
    rankloss = y_pred_score - max_prob_neg_class
    # only keep losses for positive class
    rankloss = rankloss * y_true
    # only keep losses where score is below the max
    rankloss = tf.square(tf.clip_by_value(rankloss, -10, 0))
    # average the loss for just the positive outcomes:
    rankloss = tf.reduce_sum(rankloss) / tf.reduce_sum(y_true)
    return rankloss


def rank_loss1(y_true, logits):
    # the maximum score for negative class outcome is
    max_logit_neg_class = tf.reduce_max(logits * (y_true == 0))
    # determine how much each score is above or below it
    rankloss = logits - max_logit_neg_class
    # only keep losses for positive class
    rankloss = rankloss * y_true
    # only keep losses where score is below the max
    rankloss = tf.square(tf.clip_by_value(rankloss, -10, 0))
    # average the loss for just the positive outcomes:
    rankloss = tf.reduce_sum(rankloss) / tf.reduce_sum(y_true)
    return rankloss


def rank_net_loss(y_true, logits):
    '''
    assume there are k1 negative samples (N1, .., Nk1) and k2 positive samples in the batch.
    for each positive samples(P1), build a vector of size k1, 
    element i in this vector  is Delta =  logit(P1) - logit(Ni), i=1, .., k1. if this value is negative
    it means the negative sample is scored higher than positive one -> incur loss!
    '''
    y_true = tf.cast(y_true, tf.int32)
    parts = tf.dynamic_partition(logits, y_true, 2)
    score_neg = parts[0]
    score_pos = parts[1]
    score_neg = tf.expand_dims(score_neg, axis=-1)
    score_pos = tf.expand_dims(score_pos, axis=0)
    rank_loss = score_pos - score_neg
    rank_loss *= 1
    rank_loss = tf.nn.sigmoid(-rank_loss)
    return tf.reduce_mean(rank_loss)

