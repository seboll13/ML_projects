# 2020 EPFL Machine Learning - Project 1 structure

### Python scripts
- The script ```run.py``` contains all methods we have implemented and optimized, but executes only the one that gave us the best F1-score (10-fold cross-validation ridge regression)
- We can however provide an other argument in order for it to execute the desired algorithm: 
```shell
python run.py algorithm_name
```
where ```algorithm_name``` can be: 
1. ```ls_GD``` for Gradient descent returning the loss and weights of the minimal loss index
2. ```ls_SGD``` for Stochastic gradient descent also returning the loss and weights of the minimal loss index
3. ```ls``` for Least squares regression using the normal equations
4. ```r_reg``` for Ridge regression with k-fold cross-validation over a set of lambdas
5. ```log_reg``` for Logistic regression with cross-validation over a set of gammas, for 2000 iterations
6. ```reg_log_reg``` for Regularized Logistic regression with cross-validation over a set of lambdas, and gamma of 6e-5, for 2000 iterations
- We also have included the implementations script ```implementations.py``` which both contain all methods implemented overall.
- **Note: it is important to put the datasets in the same folder as the one that contains the scripts.**

### Documentation
- All functions we have written are documented but some more in-depth details are given in the report
- The report contains all the work we have done regarding data cleaning and standardization as well as more technical parts of our algorithms and the reasons we have made these choices.