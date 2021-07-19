# implementing a sigmoid activation function
def sigmoid(z):
    s = 1.0/ (1 + np.exp(-z))    
    return s