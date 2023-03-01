import numpy as np
import tensorflow as tf

embedding_dim = 5
n_class = 1
num_iterations=100000
learning_rate=0.005

def Multilearn_GLRM_Semisupervised(A, X, Y, n_class):
    """
    A, X, Y original matrices
    
    A: n x d
    X: n x p  where p < d
    Y: d x p

    A', X', Y' are augmented matrices

    A': Append the target variable before the first column 
      : n x (1 + p)
    
    X' : Append 0 before the first column of X and A after the last column of X. Lets call it X'
       : Append a column of 1s after the last column of X'
       : n x (1 + d + p + 1)

    Y' : Append row of 0 before the first row of Y. Let's call it Y'
       : Append matrix 0 (p+1 x  p) after the last row of Y'
       : Append a column of random values before the first column of Y'
       : Set Y'[0, 0] = 0
       : (1 + d + p + 1) * (1 + p)

    X' = [0     X      A     1]
         [      ........      ]
         [      ........      ]

    Y' = [0               0   ]
         [Beta_for_X      Y   ]
         [    .............   ]
         [Beta_for_A      0   ]
         [    .............   ]
    
    A' = [Target    Covariate ]
         [    .............   ]
         [    .............   ]
         [    .............   ]

    Loss(X, Y) = T(L1, L2) + reg_x(X) + reg_y(Y)

    T(L1, L2) = L1 + L2

    L1 = A'[0:, 1:] - X'[0:, 1:] * Y'[1:, 1: #rows cols in Y] where * is matrix multiplication
    L2 = binary_cross_entropy(A[0:, 0], exp(X[0:, 1:] * Y[1:, 0])/(1 + exp(X[0:, 1:] * Y[1:, 0])))
    
    reg_X: L2 penalty on X'[0:, 1:#row in X] 
         : inf on X'[0:,0] -> To impose all values in col0 = 0 
         : 0 on X'[0:, 1 + #row in X:] -> To impose no penalty

    reg_Y: L1 penalty on Y'[1:, 0] -> induce sparsity in regression
         : L2 on Y'[1:d+1,1:p+1]
         : inf on Y'[0, :] & Y'[d+1:, 1:p+1]-> To impose all values in col0 = 0 

    """

    semisupervised_params = {
        "A_start_row": 0,
        "A_end_row" : A.shape[0],
        "A_start_col" : 0,
        "A_end_col" : n_class,

        "X_start_row": 0,
        "X_end_row" : X.shape[0],
        "X_start_col" : 1,
        "X_end_col" : X.shape[1],

        "Y_start_row": 1,
        "Y_end_row" : Y.shape[0],
        "Y_start_col" : 0,
        "Y_end_col" : n_class,

        "loss": 'MSE',
        "weight": 0.3
    }

    matrix_factorization_params = {

        "A_start_row": 0,
        "A_end_row" : A.shape[0],
        "A_start_col" : n_class,
        "A_end_col" : A.shape[1],

        "X_start_row": 0,
        "X_end_row" : X.shape[0],
        "X_start_col" : 1,
        "X_end_col" : X.shape[1]-A.shape[1]+n_class-1,

        "Y_start_row": 1,
        "Y_end_row" : Y.shape[0]-A.shape[1]+n_class-1,
        "Y_start_col" : n_class,
        "Y_end_col" : Y.shape[1],
        "weight": 0.7,
        
        "loss": 'MAE'
    }

    regularize_X_condition_1 ={
        "X_start_row": 0,
        "X_end_row" : X.shape[0],
        "X_start_col" : 0,
        "X_end_col" : 1,
        "penalty_type" : 'inf',
        "alpha": 99999999999.9
    }

    regularize_X_condition_2 ={
        "X_start_row": 0,
        "X_end_row" : X.shape[0],
        "X_start_col" : X.shape[1]-A.shape[1]+n_class-1,
        "X_end_col" : X.shape[1],
        "penalty_type" : 'constant',
        "constant": 1,
        "alpha": 0
    }

    regularize_X_condition_3 ={
        "X_start_row": 0,
        "X_end_row" : X.shape[0],
        "X_start_col" : 1,
        "X_end_col" : X.shape[1]-A.shape[1]+n_class-1,
        "penalty_type" : 'L2',
        "alpha": 0.005
    }

    regularize_X_condition_4 ={
        "X_start_row": 0,
        "X_end_row" : X.shape[0],
        "X_start_col" : 1,
        "X_end_col" : X.shape[1]-A.shape[1]+n_class-1,
        "penalty_type" : 'L1',
        "alpha": 0.0
    }

    regularize_Y_condition_1 ={
        "X_start_row": 1,
        "X_end_row" : Y.shape[0]-1,
        "X_start_col" : 0,
        "X_end_col" : n_class,
        "penalty_type" : 'L2',
        "alpha": 0.0005
    }
    regularize_Y_condition_1 ={
        "X_start_row": 1,
        "X_end_row" : Y.shape[0]-1,
        "X_start_col" : 0,
        "X_end_col" : n_class,
        "penalty_type" : 'L1',
        "alpha": 0.0005
    }

    regularize_Y_condition_2 ={
        "X_start_row": 1,
        "X_end_row" : Y.shape[0]-A.shape[1]+n_class-1,
        "X_start_col" : n_class,
        "X_end_col" : Y.shape[1],
        "penalty_type" : 'L2',
        "alpha": 0.0005
    }

    regularize_Y_condition_3 ={
        "X_start_row": 0,
        "X_end_row" : 1,
        "X_start_col" : 0,
        "X_end_col" : Y.shape[1],
        "penalty_type" : 'inf',
        "alpha": 999999999.9
    }

    regularize_Y_condition_4 ={
        "X_start_row": Y.shape[0]-A.shape[1]+n_class-1,
        "X_end_row" : Y.shape[0],
        "X_start_col" : n_class,
        "X_end_col" : Y.shape[1],
        "penalty_type" : 'inf',
        "alpha": 999999999.9
    }

    X_regularization_loss_list = [regularize_X_condition_1, regularize_X_condition_2, regularize_X_condition_3, regularize_X_condition_4]
    Y_regularization_loss_list = [regularize_Y_condition_1, regularize_Y_condition_2, regularize_Y_condition_3, regularize_Y_condition_4]
    GLRM_loss_list = [semisupervised_params, matrix_factorization_params]

    def update_grad(X, n_class):
        return tf.Variable(np.hstack(( X[:,0:X.shape[1]-A.shape[1]+n_class-1], A[:,n_class:] ,np.ones(shape=(X.shape[0], 1)) )))

    X_grad_restrictions =  update_grad
    Y_grad_restrictions =  None

    return [GLRM_loss_list, X_regularization_loss_list, Y_regularization_loss_list, X_grad_restrictions, Y_grad_restrictions]