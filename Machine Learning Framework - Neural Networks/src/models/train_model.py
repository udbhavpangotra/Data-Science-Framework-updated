def neural_network(X, Y, n_h, num_iterations=100):
    n_x = network_architecture(X, Y)[0]
    n_y = network_architecture(X, Y)[2]
    
    params = define_network_parameters(n_x, n_h, n_y)
    for i in range(0, num_iterations):
        results = forward_propagation(X, params)
        error = compute_error(results['A2'], Y)
        derivatives = backward_propagation(params, results, X, Y) 
        params = update_parameters(params, derivatives)    
    return params


y = Y.values.reshape(1, Y.size)
x = X.T.as_matrix()
model = neural_network(x, y, n_h = 10, num_iterations = 10)