import tensorflow as tf
import numpy as np
import gc

def get_L1_penalty(X, alpha):
    return alpha*tf.reduce_sum(tf.abs(X))

def get_L2_penalty(X, alpha):
    return alpha*tf.reduce_sum(tf.square(X))

def get_regularization_loss(X, loss_info):
    penalty = loss_info['penalty_type']
    alpha = loss_info['alpha']
    X_00, X_10, X_01, X_11 = loss_info['X_start_row'], loss_info['X_end_row'], loss_info['X_start_col'], loss_info['X_end_col']

    if penalty == 'L2':
        loss_regularization_X = get_L2_penalty(X[X_00:X_10, X_01:X_11], alpha)
    elif penalty == 'L1':
        loss_regularization_X = get_L1_penalty(X[X_00:X_10, X_01:X_11], alpha)
    elif penalty == 'inf':
        loss_regularization_X = 999999999*tf.reduce_sum(tf.abs(X[X_00:X_10, X_01:X_11]))
    else:
        loss_regularization_X = tf.Variable(0, dtype=tf.float64)

    return loss_regularization_X


def construct_loss_function(A, X, Y, loss_info):
    A_00, A_10, A_01, A_11 = loss_info['A_start_row'], loss_info['A_end_row'], loss_info['A_start_col'], loss_info['A_end_col']
    X_00, X_10, X_01, X_11 = loss_info['X_start_row'], loss_info['X_end_row'], loss_info['X_start_col'], loss_info['X_end_col']
    Y_00, Y_10, Y_01, Y_11 = loss_info['Y_start_row'], loss_info['Y_end_row'], loss_info['Y_start_col'], loss_info['Y_end_col']

    
    loss_name = loss_info['loss']
    if loss_name == 'MSE':
        loss = tf.reduce_mean(tf.square(A[A_00:A_10, A_01:A_11] - tf.matmul(X[X_00:X_10, X_01:X_11],Y[Y_00:Y_10, Y_01:Y_11])))
    elif loss_name == 'MAE':
        loss = tf.reduce_mean(tf.abs(A[A_00:A_10, A_01:A_11] - tf.matmul(X[X_00:X_10, X_01:X_11],Y[Y_00:Y_10, Y_01:Y_11])))
    elif loss_name == 'MAPE':
        loss = tf.reduce_mean(tf.keras.losses.mean_absolute_percentage_error(A[A_00:A_10, A_01:A_11], tf.matmul(X[X_00:X_10, X_01:X_11],Y[Y_00:Y_10, Y_01:Y_11])))
    elif loss_name == 'kl_divergence':
        loss = tf.reduce_mean(tf.keras.losses.kullback_leibler_divergence(A[A_00:A_10, A_01:A_11], tf.matmul(X[X_00:X_10, X_01:X_11],Y[Y_00:Y_10, Y_01:Y_11])))
    elif loss_name == 'binary_crossentropy':
        sigmoid = tf.keras.activations.sigmoid(tf.matmul(X[X_00:X_10, X_01:X_11],Y[Y_00:Y_10, Y_01:Y_11]))
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(A[A_00:A_10, A_01:A_11], sigmoid))
    elif loss_name == 'hinge':
        softmax = tf.nn.softmax(tf.matmul(X[X_00:X_10, X_01:X_11],Y[Y_00:Y_10, Y_01:Y_11]))
        loss = tf.reduce_mean(tf.keras.losses.hinge(A[A_00:A_10, A_01:A_11], softmax))
    elif loss_name == 'categorical_crossentropy':
        softmax = tf.nn.softmax(tf.matmul(X[X_00:X_10, X_01:X_11],Y[Y_00:Y_10, Y_01:Y_11]))
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(A[A_00:A_10, A_01:A_11], softmax))
    else:
        loss = tf.Variable(0, dtype=tf.float64)
    return loss


def get_gradients(A, X, Y, GLRM_loss_list, X_regulariation_list, Y_regulariation_list, functional_loss='WLSE'):
    GLRM_loss = 0
    functionla_loss_list = []
    regularization_X_loss = 0
    regularization_Y_loss = 0

    task_loss ={}
    for loss_info in GLRM_loss_list:
        task_loss[loss_info['Name']] = []
    
    task_loss[functional_loss] = []

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(X)
        tape.watch(Y)
        
        for loss_info in GLRM_loss_list:
            if functional_loss == 'weighted_average':
                weight = loss_info['weight']
                construction_loss = weight*construct_loss_function(A, X, Y, loss_info)
                GLRM_loss += construction_loss
                functionla_loss_list.append(construction_loss.numpy())
                task_loss[loss_info['Name']].append(construction_loss.numpy())
            
            elif functional_loss == 'max':
                temp_loss = construct_loss_function(A, X, Y, loss_info)
                if temp_loss > GLRM_loss:
                    GLRM_loss = temp_loss
                functionla_loss_list.append(temp_loss.numpy())
                task_loss[loss_info['Name']].append(temp_loss.numpy())
            
            elif functional_loss == 'WLSE':
                weight = loss_info['weight']
                alpha = loss_info['alpha']
                construction_loss = construct_loss_function(A, X, Y, loss_info)
                GLRM_loss += weight*tf.exp(alpha*construction_loss)
                functionla_loss_list.append(construction_loss.numpy())

                task_loss[loss_info['Name']].append(construction_loss.numpy())

            else:
                construction_loss = construct_loss_function(A, X, Y, loss_info)
                GLRM_loss += construction_loss
                functionla_loss_list.append(construction_loss.numpy())
        
        if functional_loss == 'WLSE':
            alpha = loss_info['alpha']
            GLRM_loss = tf.math.log(GLRM_loss)/alpha
        task_loss[functional_loss].append(GLRM_loss.numpy())

        for loss_info in X_regulariation_list:
            regularization_X_loss += (regularization_X_loss+get_regularization_loss(X, loss_info))

        for loss_info in Y_regulariation_list:
            regularization_Y_loss +=  (regularization_Y_loss+get_regularization_loss(Y, loss_info))
        
        loss = regularization_X_loss + regularization_Y_loss + GLRM_loss

    return_dictionary = {
        'total_loss': loss,
        'loss_matrix_factorization': GLRM_loss,
        # 'loss_regularization_X': regularization_X_loss,
        # 'loss_regularization_Y': regularization_Y_loss,
        'gradients': tape.gradient(loss, {'X': X, 'Y': Y}),
        'functional_loss': functionla_loss_list,
        'task_loss': task_loss
    }
    return return_dictionary

def alternating_minimization(A, X, Y, GLRM_loss_list, X_regulariation_list, Y_regulariation_list, X_grad_restrictions=None, Y_grad_restrictions=None, num_iterations=1000000, learning_rate=0.00005, n_class=0, functional_loss='WLSE', tol=1e-5):
    prev_epoch_loss = +999999999999
    
    best_X = tf.Variable(X)
    best_Y = tf.Variable(Y)
    best_epoch_loss = 999999999999
    
    task_loss ={}
    for loss_info in GLRM_loss_list:
        task_loss[loss_info['Name']] = []
    task_loss[functional_loss] = []


    for epoch in range(0, 100):
        opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
        print('epoch:', epoch)
        epoch_loss = 0
        prev_grad_loss = +999999999999

        for i in range(0, num_iterations):
            gradients_dictionary = get_gradients(A, X, Y, GLRM_loss_list, X_regulariation_list, Y_regulariation_list, functional_loss)
            opt.apply_gradients(zip([tf.Variable(gradients_dictionary['gradients']['X'])], [X]))

            if X_grad_restrictions is not None:
                X.assign(X_grad_restrictions(X, n_class))
            
            for key in gradients_dictionary['task_loss'].keys():
                task_loss[key].append(gradients_dictionary['task_loss'][key])

            if(i%99) == 0:
                print('iter:', i+1, 'Total loss:', gradients_dictionary['total_loss'], gradients_dictionary['functional_loss'])
            
            if abs(gradients_dictionary['total_loss'] - prev_grad_loss)/abs(epoch_loss) < 0.0005 or abs(gradients_dictionary['total_loss'] - prev_grad_loss) < tol:
                print('iter:', i+1, 'Total loss:', gradients_dictionary['total_loss'])
                break
            
            prev_grad_loss = gradients_dictionary['total_loss']
            epoch_loss = gradients_dictionary['total_loss']
            
            del gradients_dictionary
            gc.collect()
        
        prev_grad_loss = +999999999999
        
        for i in range(0, num_iterations):
            gradients_dictionary = get_gradients(A, X, Y, GLRM_loss_list, X_regulariation_list, Y_regulariation_list, functional_loss)
            opt.apply_gradients(zip([tf.Variable(gradients_dictionary['gradients']['Y'])], [Y]))

            if Y_grad_restrictions is not None:
                Y.assign(Y_grad_restrictions(Y, n_class))

            for key in gradients_dictionary['task_loss'].keys():
                task_loss[key].append(gradients_dictionary['task_loss'][key])

            if(i%99) == 0:
                print('iter:', i+1, 'Total loss:', gradients_dictionary['total_loss'])

            if abs(gradients_dictionary['total_loss'] - prev_grad_loss)/abs(epoch_loss) < 0.0005 or abs(gradients_dictionary['total_loss'] - prev_grad_loss) < tol:
                print('iter:', i+1, 'Total loss:', gradients_dictionary['total_loss'])
                break
            
            prev_grad_loss = gradients_dictionary['total_loss']
            epoch_loss = gradients_dictionary['total_loss']

            del gradients_dictionary
            gc.collect()

        if (epoch_loss < best_epoch_loss):
            best_epoch_loss = epoch_loss
            best_X.assign(X)
            best_Y.assign(Y)
        
        # print('epoch:', epoch, 'Total loss:', epoch_loss)
        if abs(epoch_loss - prev_epoch_loss)/abs(epoch_loss) < 0.0005 or abs(epoch_loss - prev_epoch_loss) < tol:
            break
        
        
        learning_rate = learning_rate*0.9
        prev_epoch_loss = epoch_loss

    print('Final loss:', epoch_loss, 'best loss:', best_epoch_loss)
    return best_X, best_Y, task_loss


def predict(A, X, Y,GLRM_loss_list, X_regulariation_list, Y_regulariation_list, num_iterations=1000000, learning_rate=0.00005, tol=10e-6):
    opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    prev_grad = +999999999999
    for i in range(0, num_iterations):
        gradients_dictionary = get_gradients(A, X, Y, GLRM_loss_list, X_regulariation_list, Y_regulariation_list)
        opt.apply_gradients(zip([tf.Variable(gradients_dictionary['gradients']['X'])], [X]))
        
        # if(i%99) == 0:
        #     print('iter:', i+1, 'Total loss:', gradients_dictionary['total_loss'])
            
        if abs(prev_grad- gradients_dictionary['total_loss']) < tol:
            # print('iter:', i+1, 'Total loss:', gradients_dictionary['total_loss'])
            break
        prev_grad = gradients_dictionary['total_loss']
    
    return X, Y