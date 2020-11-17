import numpy as np

# ###############################################
#
# --- Helper functions
#
# ###############################################

def compute_loss_mse(y, tx, w):
    """Calculate the MSE."""
    e = (y - tx @ w).T
    return (e @ e.T) / (2 * e.shape[1])


def compute_gradient_mse(y, tx, w):
    """Compute the gradient."""
    e = y.reshape(-1,1) - (tx @ w)
    gradL = (-1/tx.shape[0]) * (tx.T @ e)
    return gradL


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True, seed = 1):
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
    np.random.seed(seed)
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


# ###############################################
#
# --- Linear regression (GD and SGD)
#
# ###############################################

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters 
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient_mse(y, tx, w)
        loss = compute_loss_mse(y, tx, w)
        w -= gamma * grad
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
    return (w, loss)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters
    w = initial_w
    batch_size=1
    for n_iter in range(max_iters):
        for sy, stx in batch_iter(y, tx, batch_size, 100):
            #w = w.reshape(-1, 1)
            grad = compute_gradient_mse(sy, stx, w)
            loss = compute_loss_mse(sy, stx, w)
            w -= gamma * grad
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
            bi=n_iter, ti=max_iters - 1, l=loss))
    return (w, loss)


# ###############################################
#
# --- Least squares and ridge regression
#
# ###############################################
def least_squares(y, tx):
    """calculate the least squares using normal equations"""
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    return (w, compute_loss_mse(y, tx, w))


def ridge_regression(y, tx, lambda_):
    """ridge regression implementation using normal equations"""
    m, n = tx.shape[0], tx.shape[1]
    lambda_prime = 2 * lambda_ * m * np.eye(n)
    a = tx.T @ tx + lambda_prime
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    return (w, compute_loss_mse(y, tx, w))




# ###############################################
#
# --- Helper functions (Logistic regression)
#
# ###############################################

def sigmoid(t):
    """Logistic function of the same name.
    Inputs:
        t:value or numpy matrix 
    Returns:
        The sigmoid of the value, or of each element of the matrix.
    """
    return 1 / (1 + np.exp(-t))

def calculate_loss_nll(y, tx, w):
    """Calculates the negative log-likelihood loss.
    Inputs:
        y:labels
        tx:features
        w:weights
    Returns:
        The loss for logistic regression.
    """
    pred = sigmoid(tx @ w)
    a = -y * np.log(pred)
    b = -(1-y) * np.log(1-pred)
    cost = a+b
    cost = cost.sum()
    cost /= tx.shape[0]
    return cost

def calculate_gradient(y, tx, w):
    """Computes the gradient of loss for logistic regression.
    Inputs:
        y:labels
        tx:features
        w:weights
    Returns:
        The gradient of loss for logistic regression.
    """
    return tx.T @ (sigmoid(tx @ w) - y)


def learning_by_gradient_descent(y, tx, w, gamma):
    """Do one step of gradient descent using logistic regression.
    Inputs:
        y:labels
        tx:features
        w:weights
        gamma:learning rate
    Returns: 
        The loss and the updated w.
    """
    # compute the loss
    loss = calculate_loss_nll(y,tx,w)
    # compute the gradient
    grad = calculate_gradient(y,tx,w)
    grad /= tx.shape[1]
    # update w
    w -= gamma * grad
    return loss, w


# ###############################################
#
# --- Logistic_regression
#
# ###############################################

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Runs the logistic regression algorithm, for either the specified amount of iterations or until the loss converges under a set threshold.
    Inputs:
        y:labels
        tx:features
        initial_w:starting weight vector
        max_iters:Maximum number of iterations for which the algorithm will run
        gamma:learning rate
    Returns: 
        The weights and loss from the last iteration of the algorithm
    """
    threshold = 1e-8
    #List of losses at each iterations
    losses = []
    #List of average losses for each 100 iterations
    avg_losses = []
    w = initial_w
    avg_loss = 0
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        losses.append(loss)
        avg_loss += loss
        # log info and computes the average loss and weights
        if iter % 100 == 0:
            avg_loss /= 100
            print("Current iteration={i}, loss={l}, average loss since last update={al}".format(i=iter, l=loss, al=avg_loss))
            avg_losses.append(avg_loss)
            avg_loss = 0
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            print("Converged !")
            return (w,loss)
    return (w,loss)

    
    
    
# ###############################################
#
# --- Helper functions (Regularized Logistic Regression)
#
# ###############################################

"""Computes the loss and gradient of the regularized logistic regression
Inputs:
    y:labels
    tx:features
    w:weights
    lambda_:penalization rate
Returns:
    Loss and gradient
"""
def penalized_logistic_regression(y, tx, w, lambda_):
    reg_loss = ((lambda_/(2*tx.shape[0])) * np.square(np.linalg.norm(w)))
    loss = calculate_loss_nll(y, tx, w) + reg_loss
    reg_grad = lambda_ * w
    grad = calculate_gradient(y, tx, w) + reg_grad
    return loss, grad

"""Do one step of gradient descent, using the penalized logistic regression.
Inputs:
    y:labels
    tx:features
    w:weights
    gamma:learning rate
    lambda_:penaliation rate
Returns: 
    The loss and the updated w.
    """
def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    loss, grad = penalized_logistic_regression(y,tx,w,lambda_)
    grad /= tx.shape[1]
    w -= gamma * grad
    return loss, w


# ###############################################
#
# --- Regularized Logistic Regression
#
# ###############################################

"""Runs the regularized logistic regression algorithm, for either the specified amount of iterations or until the loss converges under a set threshold.
Inputs:
    y:labels
    tx:features
    lambda_:penalization rate
    initial_w:starting weight vector
    max_iters:Maximum number of iterations for which the algorithm will run
    gamma:learning rate
Returns: 
    The weights and loss from the last iteration of the algorithm
"""
def reg_logistic_regression(y, tx,lambda_, initial_w, max_iters, gamma):
    threshold = 1e-8
    #List of losses at each iterations
    losses = []
    #List of average losses for each 100 iterations
    avg_losses = []
    w = initial_w
    avg_loss = 0
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        losses.append(loss)
        avg_loss += loss
        # log info and computes the average loss and weights
        if iter % 100 == 0:
            avg_loss /= 100
            print("Current iteration={i}, loss={l}, average loss since last update={al}".format(i=iter, l=loss, al=avg_loss))
            avg_losses.append(avg_loss)
            avg_loss = 0
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            print("Converged !")
            return (w,loss)
    return (w,loss)

