import datetime
import numpy as np
import matplotlib.pyplot as plt

from proj1_helpers import *

# ###############################################
#
# --- Data loading
#
# ###############################################
DATA_TRAIN_PATH = '../data/train.csv'
y, x, ids = load_csv_data(DATA_TRAIN_PATH)

# ###############################################
#
# --- Helper functions for Linear Regression
#
# ###############################################
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def compute_loss(y, tx, w):
    """Calculate the MSE."""
    e = y.reshape(-1,1) - (tx @ w)
    return np.square(e).mean()/2


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y.reshape(-1,1) - (tx @ w)
    gradL = (-1/tx.shape[0]) * (tx.T @ e)
    return gradL


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        print(n_iter)
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * grad
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws


def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    num_batches = 100#int(len(y) / batch_size)
    for n_iter in range(max_iters):
        for sy, stx in batch_iter(y, tx, batch_size, int(tx.shape[0]/batch_size)):
            #w = w.reshape(-1, 1)
            grad = compute_gradient(sy, stx, w)
            loss = compute_loss(sy, stx, w)
            w = w - gamma * grad
            # store w and loss
            ws.append(w)
            losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws


# ###############################################
#
# --- Linear Regression algorithm implementations
#
# ###############################################
# Data standardization
x, mean_x, std_x = standardize(x)
tX = np.c_[np.ones(y.shape[0]), x]

# Define the parameters of the algorithm.
max_iters = 50
gamma = 0.05
batch_size = int(tX.shape[0]/1000)
w_initial = np.zeros((tX.shape[1], 1))

# Start of SGD.
start_time = datetime.datetime.now()
sgd_losses, sgd_ws = least_squares_SGD(y, tX, w_initial, batch_size, max_iters, gamma)
end_time = datetime.datetime.now()

# Print result
execution_time = (end_time - start_time).total_seconds()
print("SGD: execution time={t:.3f} seconds".format(t=execution_time))


# ###############################################
#
# --- Linear Regression data prediction and output submission
#
# ###############################################
DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

OUTPUT_PATH = '../data/submission.csv'
y_pred = predict_labels(sgd_ws, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)