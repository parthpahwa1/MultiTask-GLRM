import numpy as np
import tensorflow as tf

decay = 0.8

def Multilearn_GLRM_Supervised_Embeddings_Train_Params(A, X, Y, n_class):
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

    X' = [0      X      1]
         [      ...      ]
         [      ...      ]

    Y' = [0               0   ]
         [Beta_for_X      Y   ]
         [    .............   ]
         [Beta_0          0   ]
    
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
        'Name': 'SupervisedEmbeddings',
        "Description": "Supervised Learning Matrix Location",
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

        "loss": 'categorical_crossentropy',
        "weight": 0.5,
        "alpha": 1e-7
    }

    matrix_factorization_params = {
        'Name': 'Factorization',
        "Description": "Matrix Factorization Matrix Location",

        "A_start_row": 0,
        "A_end_row" : A.shape[0],
        "A_start_col" : n_class,
        "A_end_col" : A.shape[1],

        "X_start_row": 0,
        "X_end_row" : X.shape[0],
        "X_start_col" : 1,
        "X_end_col" : X.shape[1],

        "Y_start_row": 1,
        "Y_end_row" : Y.shape[0],
        "Y_start_col" : n_class,
        "Y_end_col" : Y.shape[1],
        "weight": 0.5,
        "alpha": 1e-7,
        "loss": 'MSE'
    }

    regularize_X_condition_1 ={
        "Description": "Regularize first column of X (constant 0)",

        "X_start_row": 0,
        "X_end_row" : X.shape[0],
        "X_start_col" : 0,
        "X_end_col" : 1,
        "penalty_type" : 'inf',
        "alpha": 99999999999.9
    }


    regularize_X_condition_2 ={
        "Description": "Regularize Embeddings in X",
        "X_start_row": 0,
        "X_end_row" : X.shape[0],
        "X_start_col" : 1,
        "X_end_col" : X.shape[1],
        "penalty_type" : 'L2',
        "alpha": 0.00005
    }

    regularize_X_condition_3 ={
        "Description": "Regularize Embeddings in X",
        "X_start_row": 0,
        "X_end_row" : X.shape[0],
        "X_start_col" : 1,
        "X_end_col" : X.shape[1],
        "penalty_type" : 'L1',
        "alpha": 0.00001
    }

    regularize_Y_condition_1 ={
        "Description": "Regularize parameter estimate for Beta for embeddings in Y",
        "X_start_row": 1,
        "X_end_row" : Y.shape[0]-1,
        "X_start_col" : 0,
        "X_end_col" : n_class,
        "penalty_type" : 'L2',
        "alpha": 0.0005
    }
    regularize_Y_condition_2 ={
        "Description": "Regularize parameter estimate for Beta for embeddings in Y",
        "X_start_row": 1,
        "X_end_row" : Y.shape[0]-1,
        "X_start_col" : 0,
        "X_end_col" : n_class,
        "penalty_type" : 'L1',
        "alpha": 0.0003
    }


    regularize_Y_condition_3 ={
        "Description": "Regularize Embedding Matrix in Y",
        "X_start_row": 1,
        "X_end_row" : Y.shape[0]-1,
        "X_start_col" : n_class,
        "X_end_col" : Y.shape[1],
        "penalty_type" : 'L2',
        "alpha": 0.00005
    }

    regularize_Y_condition_4 ={
        "Description": "Regularize Embedding Matrix in Y",
        "X_start_row": 1,
        "X_end_row" : Y.shape[0]-1,
        "X_start_col" : n_class,
        "X_end_col" : Y.shape[1],
        "penalty_type" : 'L1',
        "alpha": 0.00001
    }

    regularize_Y_condition_5 ={
        "Description": "Regularize first 0 row in Y",
        "X_start_row": 0,
        "X_end_row" : 1,
        "X_start_col" : 0,
        "X_end_col" : Y.shape[1],
        "penalty_type" : 'inf',
        "alpha": 999999999.9
    }


    X_regularization_loss_list = [regularize_X_condition_1, regularize_X_condition_2, regularize_X_condition_3]
    Y_regularization_loss_list = [regularize_Y_condition_1, regularize_Y_condition_2, regularize_Y_condition_3, regularize_Y_condition_4, regularize_Y_condition_5]
    GLRM_loss_list = [semisupervised_params, matrix_factorization_params]

    def update_grad_X(X, n_class):
        return tf.Variable(np.hstack(( X[:,0:X.shape[1]-1], np.ones(shape=(X.shape[0], 1)) )))

    def update_grad_Y(Y, n_class):
        temp = Y[:-1,n_class:]
        temp = np.vstack((temp, np.zeros(shape=(1, temp.shape[1]))))
        temp = np.hstack((Y[:,:n_class], temp))
        return tf.Variable(temp)

    X_grad_restrictions =  update_grad_X
    Y_grad_restrictions =  update_grad_Y

    return [GLRM_loss_list, X_regularization_loss_list, Y_regularization_loss_list, X_grad_restrictions, Y_grad_restrictions]

def Multilearn_GLRM_Supervised_Embeddings_Test_Params(A, X, Y):
    GLRM_loss_list = [{
        'Name': 'Factorization',

        "A_start_row": 0,
        "A_end_row" : A.shape[0],
        "A_start_col" : 0,
        "A_end_col" : A.shape[1],

        "X_start_row": 0,
        "X_end_row" : X.shape[0],
        "X_start_col" : 0,
        "X_end_col" : X.shape[1],

        "Y_start_row": 0,
        "Y_end_row" : Y.shape[0],
        "Y_start_col" : 0,
        "Y_end_col" : Y.shape[1],
        "weight": 1,
        "alpha": 1e-7,
        "loss": 'MSE'
    }]

    X_regularization_loss_list = [
        {
            "Description": "Regularize Embeddings in X",
            "X_start_row": 0,
            "X_end_row" : X.shape[0],
            "X_start_col" : 0,
            "X_end_col" : X.shape[1],
            "penalty_type" : 'L2',
            "alpha": 0.00005
        },
        {
            "Description": "Regularize Embeddings in X",
            "X_start_row": 0,
            "X_end_row" : X.shape[0],
            "X_start_col" : 0,
            "X_end_col" : X.shape[1],
            "penalty_type" : 'L1',
            "alpha": 0.00001
        }
    ]

    Y_regularization_loss_list = []
    Y_grad_restrictions =  None
    X_grad_restrictions = None

    return [GLRM_loss_list, X_regularization_loss_list, Y_regularization_loss_list, X_grad_restrictions, Y_grad_restrictions]
