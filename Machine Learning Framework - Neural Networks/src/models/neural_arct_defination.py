def network_architecture(X, Y):
    # nodes in input layer
    n_x = X.shape[0] 
    # nodes in hidden layer
    n_h = 10          
    # nodes in output layer
    n_y = Y.shape[0] 
    return (n_x, n_h, n_y)