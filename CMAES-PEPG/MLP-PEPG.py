import numpy as np
from matplotlib import pyplot as plt
import cma
from es import SimpleGA, CMAES, PEPG, OpenES
import random
random.seed(12345)
np.random.seed(12345)
from tensorflow.examples.tutorials.mnist import  input_data
mnist = input_data.read_data_sets('/mnist', one_hot=False)
train, validation, test = mnist
train_X, train_Y = train.images[:1000], train.labels[:1000]
validation_X, validation_Y = validation.images, validation.labels
print(train_X.shape, train_Y.shape)
n_feats = train_X.shape[1]
hidden1_size = 8
hidden2_size = 8
MAX_ITERATION = 1000
NPOPULATION = 501
n_class  = 10
l2Reg = 0.0
def stable_softmax(logits):
    # logits is matrix of n*10- 10: number of classes
    max_logits_per_row = np.max(logits, axis=1)
    max_logits_per_row = max_logits_per_row.reshape((len(max_logits_per_row), 1))
    logits = logits - max_logits_per_row
    exps = np.exp(logits)
    probs = exps/np.sum(exps, axis=1).reshape((len(exps), 1))
    return probs

def crossEntropyLoss(logits,labels):
    #logits: n*10 and labels: a number: 0, 1, ..., 9
    m = labels.shape[0]
    p = stable_softmax(logits)
    log_likelihood = -np.log(p[range(m), labels])
    loss = np.sum(log_likelihood)/ m
    return loss

def glorot_normal_weight_initializer(shape):
    ''' Use for tanh activation  Glorot et al. 2012'''
    initial = np.random.randn(shape[0], shape[1])* np.sqrt(3. / (shape[0] + shape[1]))
    return initial
def weight_flatten(weights, biases):
    ''' parameters must be fed in using 1D vector'''
    weight_keys = ['h1','h2', 'out']
    b_keys = ['b1','b2', 'out']
    w = weights[weight_keys[0]].flatten()
    b = biases[b_keys[0]]
    for item in weight_keys[1:]:
        w = np.append(w, weights[item].flatten(), axis = 0)
    for item in b_keys[1:]:
        b = np.append(b, biases[item].flatten(), axis = 0)
    flattened_params = np.append(w,b, axis = 0)
    return flattened_params

def weight_shape(flat_params, n_feats, size1, size2, n_class):
    '''returns weights and biases in a dict!'''
    weights = {}; biases = {}
    weights['h1'] = flat_params[:n_feats*size1].reshape((n_feats, size1))
    weights['h2'] = flat_params[n_feats*size1: n_feats*size1+ size1*size2].reshape((size1, size2))
    weights['out'] = flat_params[n_feats*size1+ size1*size2: n_feats*size1+ size1*size2 +size2*n_class].reshape(size2, n_class)
    W_end = n_feats*size1+ size1*size2 +size2*n_class
    biases['b1']  = flat_params[W_end: W_end+size1]
    biases['b2'] = flat_params[W_end+size1: W_end+size1 + size2]
    biases['out'] = flat_params[W_end+size1 + size2:]
    return weights, biases


def MLP(flatParmas, x, labels, l2Reg):
    weights, biases = weight_shape(flatParmas, n_feats, hidden1_size, hidden2_size, n_class)

    layer_1 = np.matmul(x, weights['h1'])+ biases['b1']
    layer_1 = np.tanh(layer_1)
    layer_2 = np.matmul(layer_1, weights['h2']) + biases['b2']
    layer_2 = np.tanh(layer_2)
    output = np.matmul(layer_2, weights['out']) + biases['out']
    loss = crossEntropyLoss(output, labels)
    totWeights = len(flatParmas)
    l2Loss = sum([p**2 for p in flatParmas])
    l2Loss = l2Reg * l2Loss /totWeights
    totalLoss = loss + l2Loss # make sure CMAES is minimization if not change sign accordingly in optimization!
    return totalLoss


def MLP_predict(x, flatParmas):
    weights, biases = weight_shape(flatParmas, n_feats, hidden1_size, hidden2_size, n_class)
    layer_1 = np.matmul(x, weights['h1'])+ biases['b1']
    layer_1 = np.tanh(layer_1)
    layer_2 = np.matmul(layer_1, weights['h2']) + biases['b2']
    layer_2 = np.tanh(layer_2)
    output = np.matmul(layer_2, weights['out']) + biases['out']
    preds = np.argmax(output, axis =1)
    return preds
def accuracy(x, labels, flatParams):
    predictions = MLP_predict(x, flatParams)
    is_equal = np.equal(predictions, labels)
    is_equal = is_equal.astype(float)
    acc = np.mean(is_equal)
    return acc


# defines a function to use solver to solve fit_func
def test_solver(solver):
  history_cross_entropy = []
  accu_hist = []
  validation_acc_hist = []
  for j in range(MAX_ITERATION):
    solutions = solver.ask()
    fitness_list = np.zeros(solver.popsize)
    for i in range(solver.popsize):
      fitness_list[i] = - MLP(solutions[i], train_X, train_Y, l2Reg) # negative for maximization!
    solver.tell(fitness_list)
    result = solver.result() # first element is the best solution, second element is the best fitness
    history_cross_entropy.append(result[1])
    current_accu =accuracy(train_X, train_Y,result[0])
    val_acc =accuracy(validation_X, validation_Y, result[0])
    validation_acc_hist.append(val_acc)
    accu_hist.append(current_accu)
    if (j+1) % 100 == 0:
      print("Iteration", (j+1), 'Cross-entropy= ', round(-result[1],3), 'Accuracy= ', round(current_accu, 3))

  print("local optimum discovered by solver:\n", result[0])
  print("fitness score at this local optimum:", result[1])
  return history_cross_entropy, accu_hist,validation_acc_hist, result[0]

# there is no need to initialize parameters! I just used the parameter dict as I define in tensorflow
# only the total number of parameters and shape of each weight matrix is needed!
weights = {
            'h1': glorot_normal_weight_initializer([n_feats, hidden1_size]),
            'h2': glorot_normal_weight_initializer([hidden1_size, hidden2_size]),
            'out': glorot_normal_weight_initializer([hidden2_size, n_class])
        }
biases = {
            'b1': np.zeros(hidden1_size),
            'b2': np.zeros(hidden2_size),
            'out': np.zeros(n_class)
        }

flattened_params = weight_flatten(weights, biases)
# pred_sample = MLP_predict(train_X, flattened_params)
# print('pre optimization Train data Predicitons= ')
# print(pred_sample)
sample_cross_entropy_loss = MLP(flattened_params, train_X, train_Y, l2Reg)
print('Pre-optimization Train data cross entropy loss and accuracy=')
print(sample_cross_entropy_loss, accuracy(train_X, train_Y, flattened_params))

NPARAMS = len(flattened_params)
pepg = PEPG(NPARAMS,                         # number of model parameters
            sigma_init=0.5,                  # initial standard deviation
            learning_rate=0.1,               # learning rate for standard deviation
            learning_rate_decay=1.0,       # don't anneal the learning rate
            popsize=NPOPULATION,             # population size
            average_baseline=False,          # set baseline to average of batch
            weight_decay=0.00,            # weight decay coefficient
            rank_fitness=False,           # use rank rather than fitness numbers
            forget_best=False)            # don't keep the historical best solution)

#remove the comment from next line if you want to do the PEPG
print('='*100)
print('='*45+'PEPG'+'='*45)
pepg_history, accu_hist,validation_acc_hist, best_solution = test_solver(pepg)
np.save('PEPG-flattened_solution.npy',best_solution)

print('='*100)
print(len(best_solution))
train_accu=accuracy(train_X, train_Y, best_solution)
print('Train Accuracy=', train_accu)
plt.figure(figsize=(8,4), dpi=150)
plt.subplot(121)
plt.plot(-1*np.array(pepg_history))
plt.title('PEPG- MNIST')
plt.ylabel('Cross entropy loss')
plt.xlabel('Iteration')
plt.ylim(0, 2.5)
plt.subplot(122)
plt.plot(accu_hist, label='Train accuracy')
plt.plot(validation_acc_hist, label='Validation accuracy', color='red')
plt.title('PEPG- MNIST')

plt.ylabel('Accuracy')
plt.xlabel('Iteration')
plt.ylim(0,1)
plt.legend()
plt.savefig('PEPG-MNIST.png')

plt.show()