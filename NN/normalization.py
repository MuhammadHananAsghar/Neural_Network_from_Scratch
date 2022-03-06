def normalize(_input):
    # X --> Input.

    # m-> number of training examples
    # n-> number of features
    m, n = _input.shape

    # Normalizing all the n features of X.
    for i in range(n):
        _input = (_input - _input.mean(axis=0)) / _input.std(axis=0)

    return _input