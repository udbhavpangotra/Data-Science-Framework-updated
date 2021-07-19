def forward_propagation(X, params):
    Z1 = np.dot(params['W1'], X)+params['b1']
    A1 = sigmoid(Z1)

    Z2 = np.dot(params['W2'], A1)+params['b2']
    A2 = sigmoid(Z2)
    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2} 