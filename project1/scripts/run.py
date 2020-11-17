import sys
import csv
import numpy as np
import datetime
import matplotlib.pyplot as plt

# ###############################################
#
# --- Helper functions for reading and writing csv's
#
# ###############################################

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

# ###############################################
#
# --- Prepare samples
#
# ###############################################
def standardize(data):
        """Standardize the original data set."""
        # Remove all values equal to -999 (means 'no value')
        data_ = np.ma.masked_array(data, data==-999)

        mean_data = np.mean(data_, axis=0)
        std_data = np.std(data_, axis=0)
        data_ = data_ - mean_data
        data_ = data_ / std_data

        data_ = data_.filled(fill_value=0)
        return data_, mean_data, std_data

def init():
    DATA_TRAIN_PATH = 'train.csv'
    y, x, ids = load_csv_data(DATA_TRAIN_PATH)

    y = np.expand_dims(y, axis=1)
    tX, mean_x, std_x = standardize(x)
    tX = np.c_[np.ones(x.shape[0]), tX]

    return y, tX


# ###############################################
#
# --- Helper functions
#
# ###############################################
def compute_loss(y, tx, w):
    """Calculate the MSE."""
    e = (y - tx @ w).T
    return (e @ e.T) / (2 * e.shape[1])


def compute_gradient(y, tx, w):
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



def plot_train_test(train_errors, test_errors, lambdas, title):
    """
    Function for plotting test and train loss, taken from the labs
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set
    * title: titleof the plot and savefile
    """
    plt.semilogx(lambdas, train_errors, color='b', marker='*', label="Train error")
    plt.semilogx(lambdas, test_errors, color='r', marker='*', label="Test error")
    plt.xlabel("lambda")
    plt.ylabel("loss")
    plt.title(title)
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig(title)


# ###############################################
#
# --- Linear regression (GD and SGD)
#
# ###############################################

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Implements the gradient descent algorithm.
    Inputs:
        y: labels
        tx: features
        initial_w: initial weights
        max_iters: max number of iterations for the algorithm
        gamma: learning rate
    Returns:
        Loss and weights from the minimum loss index
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w -= gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    # Return loss and weights from minimum loss index
    min_loss_idx = losses.index(min(losses))
    return losses[min_loss_idx], ws[min_loss_idx]


def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """
    Implements the stochastic gradient descent algorithm.
    Inputs:
        y: labels
        tx: features
        initial_w: initial weights
        batch_size: number of data samples in batch
        max_iters: max number of iterations for the algorithm
        gamma: learning rate
    Returns:
        Loss and weights from the minimum loss index
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for sy, stx in batch_iter(y, tx, batch_size, 1):
            #w = w.reshape(-1, 1)
            grad = compute_gradient(sy, stx, w)
            loss = compute_loss(sy, stx, w)
            w -= gamma * grad
            # store w and loss
            ws.append(w)
            losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    # Return loss and weights from minimum loss index
    min_loss_idx = losses.index(min(losses))
    return losses[min_loss_idx], ws[min_loss_idx]


def exec_least_square_GD(y, tX):
    """Executes the GD algorithm"""
    # Gradient descent parameters initialization
    max_iters = 50
    gamma = 1e-2
    w_initial = np.zeros((tX.shape[1], 1))

    # Start GD
    start_time = datetime.datetime.now()
    gd_losses, gd_ws = least_squares_GD(y, tX, w_initial, max_iters, gamma)
    end_time = datetime.datetime.now()

    # Print result
    execution_time = (end_time - start_time).total_seconds()
    print("GD: execution time={t:.3f} seconds".format(t=execution_time))

    return gd_ws


def exec_least_square_SGD(y, tX):
    """Executes the SGD algorithm"""
    # Stochastic Gradient descent parameters initialization
    max_iters = 50
    gamma = 1e-3
    batch_size = 1

    # Initialization
    w_initial = np.zeros((tX.shape[1], 1))

    # Start SGD.
    start_time = datetime.datetime.now()
    sgd_losses, sgd_ws = least_squares_SGD(y, tX, w_initial, batch_size, max_iters, gamma)
    end_time = datetime.datetime.now()

    # Print result
    execution_time = (end_time - start_time).total_seconds()
    print("SGD: execution time={t:.3f} seconds".format(t=execution_time))

    return sgd_ws


# ###############################################
#
# --- Least squares
#
# ###############################################
def least_squares(y, tx):
    """
    Implements the least squares regression using normal equations
    Inputs:
        y: labels
        tx: features
    Returns:
        Loss and weights obtained from solving the linear system
    """
    a = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    return compute_loss(y, tx, w), w


def exec_least_squares(y, tX):
    """Execute least squares algorithm"""
    # Start least squares
    start_time = datetime.datetime.now()
    ls_losses, ls_ws = least_squares(y, tX)
    end_time = datetime.datetime.now()

    # Print result
    execution_time = (end_time - start_time).total_seconds()
    print(ls_losses, ls_ws)
    print("LS: execution time={t:.3f} seconds".format(t=execution_time))

    return ls_ws


# ###############################################
#
# --- Ridge regression (k-fold)
#
# ###############################################

def ridge_regression(y, tx, lambda_):
    """
    Implements the ridge regression algorithm.
    Inputs:
        y:labels
        tx:features
        lambda_:penalization rate
    Returns:
        The weights as calculated by the ridge regression algorithm.
    """
    m, n = tx.shape[0], tx.shape[1]
    a = tx.T @ tx
    np.fill_diagonal(a, a.diagonal() + lambda_ * 2 * m)
    b = tx.T @ y
    w = np.linalg.solve(a, b)
    return w


def build_k_indices(y, k_fold, seed):
    """
    Builds k indices for k-fold cross-validation.
    Inputs:
        y:labels
        k_fold:number of group to create
        seed:seed used for the random repartition of data points
    Returns:
        An array of indices to separate the data into k groups.
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_):
    """
    Do ridge regression on all but the k'th group, and test the result on the k'th group.
    Inputs:
        y:labels
        x:features
        k_indices:array used to split the data into the necessary groups; see build_k_indices
        lambda_:penalization rate
    Returns:
        The training loss, testing loss, and associated weights
    """
    # get k'th subgroup in test, others in train
    k_indices_train = np.delete(k_indices,k,axis=0).flatten()
    k_indices_test = k_indices[k]
    y_train = y[k_indices_train]
    x_train = x[k_indices_train]
    y_test = y[k_indices_test]
    x_test = x[k_indices_test]
    # ridge regression
    weights = ridge_regression(y_train, x_train, lambda_)
    # calculate the loss for train and test data
    loss_tr = compute_loss(y_train,x_train,weights)
    loss_te = compute_loss(y_test,x_test,weights)
    return loss_tr, loss_te, weights


def exec_ridge_reg_k_fold(y, x, k, seed=1):
    """
    Computes the ridge regression and cross validates it with k-fold cross validation for lambdas between 10e-6 and 1, and plots the training and testing losses.
    (less doesn't make a difference, and more diverges and overflows)
    Inputs:
        y:labels
        x:features
        k:number of groups for k-fold cross validation
        seed:optional, used for randomization of groups
    Returns:
        Prints the progress in the console, and it will save a graph of the losses named "ridge reg k-fold"(where k is the k given in argument)
    """
    k_fold = k
    lambdas = np.logspace(-6, 0, 20)
    print(lambdas)
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    losses_tr = []
    losses_te = []
    ws = []
    # cross validation
    for ind, lambda_ in enumerate(lambdas):
        loss_tr = 0
        loss_te = 0
        w_avg = 0
        for k in range(k_fold):
            temp_loss_tr, temp_loss_te, w = cross_validation(y,x,k_indices,k,lambda_)
            w_avg += w
            loss_tr += temp_loss_tr
            loss_te += temp_loss_te
        w_avg /= k_fold
        loss_tr /= k_fold
        loss_te /= k_fold
        ws.append(w_avg)
        losses_tr.append(loss_tr[0][0])
        losses_te.append(loss_te[0][0])
        print(" lambda={l:.6f}, Training Loss={tr:.8f}, Testing Loss={te:.8f}".format(
             l=lambda_, tr=losses_tr[ind], te=losses_te[ind]))
    minTeLossIndex = losses_te.index(min(losses_te))
    print("Best test loss:" + str(losses_te[minTeLossIndex]) + " at lambda:" + str(lambdas[minTeLossIndex]))
    plot_train_test(losses_tr,losses_te,lambdas,"ridge reg " + str(k_fold) + "-fold")

    return ws[minTeLossIndex]


# ###############################################
#
# --- Helper function for logistic regression
#
# ###############################################

def split_data(x, y, ratio, seed=1):
    """
    Splits the data points and shuffles them according to a ratio and seed, respectively.
    Inputs:
        y:labels
        x:features
        ratio:ratio of training vs validation data points that ill be returned
        seed:seed for the random shuffling
    Returns:
        4 lists of data points: training features and labels, then validation features and labels.
    """
    # set seed
    np.random.seed(seed)
    #creates a list of all the indexes from the number of points, and shuffles it
    shuffle = np.arange(x.shape[0])
    np.random.shuffle(shuffle)
    #displaces the data points according to this list
    rndm_x = x[shuffle]
    rndm_y = y[shuffle]
    #separates the points according to the ratio
    training = int(ratio * x.shape[0])
    return rndm_x[:training], rndm_y[:training], rndm_x[training:], rndm_y[training:]


def sigmoid(t):
    """
    Logistic function of the same name.
    Outputs: sigma(x)=1/(1+e^(-t))
    """
    return 1 / (1 + np.exp(-t))


def calculate_loss(y, tx, w):
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
    loss = calculate_loss(y,tx,w)
    # compute the gradient
    grad = calculate_gradient(y,tx,w)
    grad /= tx.shape[1]
    # update w
    w -= gamma * grad
    return loss, w


# ###############################################
#
# --- Logistic and regularized logistic regression
#
# ###############################################
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Runs the logistic regression algorithm, for either the specified amount of iterations or until the loss converges under a set threshold.
    Inputs:
        y:labels
        tx:features
        initial_w:starting weight vector
        max_iters:Maximum number of iterations for which the algorithm will run
        gamma:learning rate
    Returns:
        The average weights that led to the best average loss over 100 iterations, said loss, and the weights from the last iterations
    """
    threshold = 1e-8
    #List of losses at each iterations
    losses = []
    #List of average losses for each 100 iterations
    avg_losses = []
    #List of average weights for each 100 iterations
    ws = []
    w = initial_w
    avg_loss = 0
    avg_w = 0
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        losses.append(loss)
        avg_loss += loss
        avg_w += w
        # log info and computes the average loss and weights
        if iter % 100 == 0:
            avg_loss /= 100
            avg_w /= 100
            print("Current iteration={i}, loss={l}, average loss since last update={al}".format(i=iter, l=loss, al=avg_loss))
            avg_losses.append(avg_loss)
            ws.append(avg_w)
            avg_loss = 0
            avg_w = 0
        # convergence criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            print("Converged !")
            return ws[-1], losses[-1], w
    #removes the first average weights and loss because it was added at the first iteration
    avg_losses = avg_losses[1:]
    ws = ws[1:]
    #compute and return the min average loss and weights, and the weights from the last iteration
    minLossIndex = avg_losses.index(min(avg_losses))
    return ws[minLossIndex], avg_losses[minLossIndex], w


def exec_log_reg_cross_validation(y, x, ratio, seed=1):
    """
    This function computes the cross validation for the logistic regression, for gammas between 1e-5 and 4e-4
    (gammas below that makes the loss rise again, while above that it diverges and overflows)
    Inputs:
        x:features
        y:labels
        ratio:ratio of training vs validation data points for the cross validation
        seed:optional, used for the shuffling of the data
    Returns:
        Prints the progress in the console, and it will save a graph of the losses named "Logistic regression"
    """
    # define gammas
    gammas = np.arange(1e-5,4e-4,5e-5)
    print(gammas)
    # split the data, and return train and test data
    x_train,y_train,x_test,y_test = split_data(x,y,ratio,seed)
    w_initial = np.zeros((x.shape[1], 1))

    loss_tr = []
    loss_te = []
    ws = []
    for ind, gamma in enumerate(gammas):
        #runs the logistic regression
        w_initial = np.zeros((x_train.shape[1], 1))
        w, loss, last_w = logistic_regression(y_train, x_train, w_initial, 2000, gamma)
        ws.append(w)
        loss_tr.append(loss)
        loss_te.append(calculate_loss(y_test, x_test, w))
        print("proportion={p}, gamma={l:.6f}, Training loss={tr:.8f}, Testing loss={te:.8f}".format(
               p=ratio, l=gamma, tr=loss_tr[ind], te=loss_te[ind]))

    minTeLossIndex = loss_te.index(min(loss_te))
    print("Best test loss:" + str(loss_te[minTeLossIndex]) + " at gamma:" + str(gammas[minTeLossIndex]))
    # Plot the obtained results
    plot_train_test(loss_tr, loss_te, gammas, "Logistic regression")

    return ws[minTeLossIndex]


def penalized_logistic_regression(y, tx, w, lambda_):
    """Computes the loss and gradient of the regularized logistic regression
    Inputs:
        y:labels
        tx:features
        w:weights
        lambda_:penalization rate
    Returns:
        Loss and gradient
    """
    reg_loss = ((lambda_/(2*tx.shape[0])) * np.square(np.linalg.norm(w)))
    loss = calculate_loss(y, tx, w) + reg_loss
    reg_grad = lambda_ * w
    grad = calculate_gradient(y, tx, w) + reg_grad
    return loss, grad


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Inputs:
        y:labels
        tx:features
        w:weights
        gamma:learning rate
        lambda_:penaliation rate
    Returns:
        The loss and the updated w.
    """
    loss, grad = penalized_logistic_regression(y,tx,w,lambda_)
    grad /= tx.shape[1]
    w -= gamma * grad
    return loss, w


def reg_logistic_regression(y, tx,lambda_, initial_w, max_iters, gamma):
    """
    Runs the regularized logistic regression algorithm, for either the specified amount of iterations or until the loss converges under a set threshold.
    Inputs:
        y:labels
        tx:features
        lambda_:penalization rate
        initial_w:starting weight vector
        max_iters:Maximum number of iterations for which the algorithm will run
        gamma:learning rate
    Returns:
        The average weights that led to the best average loss over 100 iterations, said loss, and the weights from the last iterations
    """
    threshold = 1e-8
    #List of losses at each iterations
    losses = []
    #List of average losses for each 100 iterations
    avg_losses = []
    #List of average weights for each 100 iterations
    ws = []
    w = initial_w
    avg_loss = 0
    avg_w = 0
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        losses.append(loss)
        avg_loss += loss
        avg_w += w
        # log info and computes the average loss and weights
        if iter % 100 == 0:
            avg_loss /= 100
            avg_w /= 100
            print("Current iteration={i}, loss={l}, average loss since last update={al}".format(i=iter, l=loss, al=avg_loss))
            avg_losses.append(avg_loss)
            ws.append(avg_w)
            avg_loss = 0
            avg_w = 0
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            print("Converged !")
            return ws[-1], losses[-1], w
    #removes the first average weights and loss because it was added at the first iteration
    avg_losses = avg_losses[1:]
    ws = ws[1:]
    #compute and return the min average loss and weights, and the weights from the last iteration
    minLossIndex = avg_losses.index(min(avg_losses))
    return ws[minLossIndex], avg_losses[minLossIndex], w


def exec_reg_log_reg_cross_validation(y, x, ratio, seed=1):
    """
    This function computes the cross validation for the regularized logistic regression, for laambdas between 50 and 0.01
    (lambdas below that makes no differences, while above that it diverges and overflows)
    Inputs:
        x:features
        y:labels
        ratio:ratio of training vs validation data points for the cross validation
        seed:optional, used for the shuffling of the data
    Returns:
        Prints the progress in the console, and it will save a graph of the losses named "Regularized logistic regression"
    """
    # define lambdas
    lambdas = np.array([50,40,30,20,10,1,0.1,0.01])
    print(lambdas)
    # split the data, and return train and test data
    x_train,y_train,x_test,y_test = split_data(x,y,ratio,seed)
    w_initial = np.zeros((x.shape[1], 1))

    loss_tr = []
    loss_te = []
    ws = []
    for ind, lambda_ in enumerate(lambdas):
        #runs the regularized logistic regression
        w_initial = np.zeros((x_train.shape[1], 1))
        w, loss, last_w = reg_logistic_regression(y_train, x_train,lambda_, w_initial, 2000, 6e-5)
        ws.append(w)
        loss_tr.append(loss)
        loss_te.append(calculate_loss(y_test, x_test, w))
        print("proportion={p}, gamma={l:.6f}, Training loss={tr:.8f}, Testing loss={te:.8f}".format(
               p=ratio, l=lambda_, tr=loss_tr[ind], te=loss_te[ind]))

    minTeLossIndex = loss_te.index(min(loss_te))
    print("Best test loss:" + str(loss_te[minTeLossIndex]) + " at lambda:" + str(lambdas[minTeLossIndex]))
    # Plot the obtained results
    plot_train_test(loss_tr, loss_te, lambdas, "Regularized logistic regression")

    return ws[minTeLossIndex]


# ###############################################
#
# --- Prepare submission and write csv
#
# ###############################################
def create_submission(weights):
    DATA_TEST_PATH = 'test.csv'
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

    tX_test, mean_x_test, std_x_test = standardize(tX_test)
    tX_test = np.c_[np.ones(tX_test.shape[0]), tX_test]

    OUTPUT_PATH = 'output.csv'
    y_pred = predict_labels(weights, tX_test)
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

def create_submission_reg(weights):
    DATA_TEST_PATH = 'test.csv'
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

    tX_test, mean_x_test, std_x_test = standardize(tX_test)
    tX_test = np.c_[np.ones(tX_test.shape[0]), tX_test]

    OUTPUT_PATH = 'output.csv'
    y_pred = 2*sigmoid(tX_test @ weights) - 1

    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)


# ###############################################
#
# --- Main program
#
# ###############################################
def main():
    if len(sys.argv) >= 2:
        # Initialization
        print("Initialization")
        y, tX = init()

        if sys.argv[1] == "least_squares_GD" or sys.argv[1] == "ls_GD":
            print("Executing least_squares_GD")
            weights = exec_least_square_GD(y, tX)
            create_submission(weights)

        elif sys.argv[1] == "least_squares_SGD" or sys.argv[1] == "ls_SGD":
            print("Executing least_squares_SGD")
            weights = exec_least_square_SGD(y, tX)
            create_submission(weights)

        elif sys.argv[1] == "least_squares" or sys.argv[1] == "ls":
            print("Executing least_squares")
            weights = exec_least_squares(y, tX)
            create_submission(weights)

        elif sys.argv[1] == "ridge_regression" or sys.argv[1] == "r_reg":
            k = 10
            if len(sys.argv) == 3:
                k = int(sys.argv[2])
            print("Executing ridge_regression with k : " + str(k))
            weights = exec_ridge_reg_k_fold(y, tX, k)
            create_submission(weights)

        elif sys.argv[1] == "logistic_regression" or sys.argv[1] == "log_reg":
            y[y == -1] = 0
            ratio = 0.8
            if len(sys.argv) == 3:
                ratio = int(sys.argv[2])
            print("Executing logistic_regression with ratio : " + str(ratio))
            weights = exec_log_reg_cross_validation(y, tX, ratio)
            create_submission_reg(weights)

        elif sys.argv[1] == "regularized_logistic_regression" or sys.argv[1] == "reg_log_reg":
            y[y == -1] = 0
            ratio = 0.8
            if len(sys.argv) == 3:
                ratio = int(sys.argv[2])
            print("Executing regularized_logistic_regression with ratio : " + str(ratio))
            weights = exec_reg_log_reg_cross_validation(y, tX, ratio)
            create_submission_reg(weights)
    else:
        y[y == -1] = 0
        ratio = 0.8
        print("Executing logistic_regression with ratio : " + str(ratio))
        weights = exec_log_reg_cross_validation(y, tX, ratio)
        create_submission_reg(weights)

if __name__ == '__main__':
    main()