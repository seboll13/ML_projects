#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Useful starting lines
#get_ipython().run_line_magic('matplotlib', 'inline')
import datetime
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# ## Load the training data into feature matrix, class labels, and event ids:

# In[2]:


from proj1_helpers import *
DATA_TRAIN_PATH = '../data/train.csv'
y, x, ids = load_csv_data(DATA_TRAIN_PATH)

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

x, mean_x, std_x = standardize(x)
tX = np.c_[np.ones(x.shape[0]), x]


# ## Do your thing crazy machine learning thing here :) ...

# In[3]:


# ###############################################
#
# --- Helper functions for Linear Regression
#
# ###############################################

def compute_loss(y, tx, w):
    """Calculate the MSE."""
    e = y.reshape(-1,1) - (tx @ w)
    return np.square(e).mean()/2


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y.reshape(-1,1) - (tx @ w)
    gradL = (-1/tx.shape[0]) * (tx.T @ e)
    return gradL


# In[4]:


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * grad
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

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


def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
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


def least_squares(y, tx):
    """calculate the least squares."""
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    return compute_loss(y, tx, w), w

# In[5]:


# Linear regression parameters initialization
max_iters = 50
gamma = 1
initial_w = np.zeros((tX.shape[1], 1))

#least_squares_GD(y, tX, initial_w, max_iters, gamma)


# In[6]:


# Define the parameters of the algorithm.
max_iters = 50
gamma = 0.05
batch_size = int(tX.shape[0]/1000)

# Initialization
w_initial = np.zeros((tX.shape[1], 1))

# Start SGD.
start_time = datetime.datetime.now()
#sgd_losses, sgd_ws = least_squares_SGD(y, tX, w_initial, batch_size, max_iters, gamma)
ls_losses, ls_ws = least_squares(y, tX)
end_time = datetime.datetime.now()

# Print result
execution_time = (end_time - start_time).total_seconds()
print("Execution time={t:.3f} seconds".format(t=execution_time))

#weights = sgd_ws[-1]
weights = ls_ws


# In[7]:


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    m, n = tx.shape[0], tx.shape[1]
    lambda_prime = lambda_ * 2 * m * np.eye(n)
    a = tx.T.dot(tx) + lambda_prime
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)


# In[8]:


# ###############################################
#
# --- Helper functions for Logistic Regression
#
# ###############################################

def sigmoid(t):
    """Logistic function"""
    return 1 / (1 + np.exp(-t))


def calculate_loss(y, tx, w):
    """Negative log-likelihood"""
    pred = sigmoid(tx.dot(w))
    a = -y.T.dot(np.log(pred))
    b = (1-y).T.dot(np.log(1-pred))
    return a-b


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T.dot(sigmoid(tx.dot(w))-y)


def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    arr = sigmoid(tx.dot(w)).T[0]
    pred = np.diag(arr)
    S = np.multiply(pred, 1-pred)
    return tx.T.dot(S).dot(tx)


# In[9]:


def logistic_regression(y, tx, w):
    """return the loss, gradient, and Hessian."""
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    hess = calculate_hessian(y, tx, w)
    return loss, grad, hess


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient."""
    loss = calculate_loss(y, tx, w) + (lambda_ * np.square(np.linalg.norm(w)))
    grad = 2 * lambda_ * w + calculate_gradient(y, tx, w)
    return loss, grad


# ## Generate predictions and save ouput in csv format for submission:

# In[10]:


DATA_TEST_PATH = '../data/test.csv'
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
tX_test, mean_x_test, std_x_test = standardize(tX_test)
tX_test = np.c_[np.ones(tX_test.shape[0]), tX_test]


# In[12]:


OUTPUT_PATH = '../data/output.csv'
y_pred = predict_labels(weights, tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)


# In[1]:


# Convert this notebook to python script by executing this cell
# get_ipython().system('jupyter nbconvert --to script project1.ipynb')


# In[ ]:




