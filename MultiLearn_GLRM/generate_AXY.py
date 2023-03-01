import numpy as np
import tensorflow as tf

def get_semisupervised_glrm_train_form(input_matrix, target, num_classes=1, embedding_dim = 5):

    A = input_matrix
    A_prime = np.hstack((np.array(target), input_matrix))
    A_prime =tf.constant(A_prime)


    X = np.random.standard_normal((A.shape[0], embedding_dim))
    X_prime = np.hstack((np.zeros(shape=(A.shape[0], 1)), X))
    X_prime = np.hstack((X_prime, A))
    X_prime = np.hstack((X_prime, np.ones(shape=(A.shape[0], 1))))

    X_prime = tf.Variable(X_prime)

    Y = np.random.standard_normal((embedding_dim, A.shape[1]))
    Y_prime = np.vstack((np.zeros(shape=(1, A.shape[1])), Y))
    Y_prime = np.vstack((Y_prime, np.zeros(shape=(A.shape[1]+1, A.shape[1]))))
    Y_prime = np.hstack((np.random.standard_normal((Y_prime.shape[0], num_classes)), Y_prime))
    Y_prime[0,0:num_classes] = 0
    Y_prime = tf.Variable(Y_prime)

    return A_prime, X_prime, Y_prime


def get_supervised_embedding_glrm_train_form(input_matrix, target, num_classes=1, embedding_dim = 5):

    A = input_matrix
    A_prime = np.hstack((np.array(target), input_matrix))
    A_prime =tf.constant(A_prime)


    X = np.random.standard_normal((A.shape[0], embedding_dim))
    X_prime = np.hstack((np.zeros(shape=(A.shape[0], 1)), X))
    X_prime = np.hstack((X_prime, np.ones(shape=(A.shape[0], 1))))
    X_prime = tf.Variable(X_prime)

    Y = np.random.standard_normal((embedding_dim, A.shape[1]))
    Y_prime = np.vstack((np.zeros(shape=(1, A.shape[1])), Y))
    Y_prime = np.vstack((Y_prime, np.zeros(shape=(1, Y_prime.shape[1]))))
    Y_prime = np.hstack((np.random.standard_normal((Y_prime.shape[0], num_classes)), Y_prime))

    Y_prime[0,0:num_classes] = 0
    Y_prime = tf.Variable(Y_prime)

    return A_prime, X_prime, Y_prime
