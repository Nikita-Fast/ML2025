import numpy as np
from tqdm import tqdm
import time

from Source.dataset import load_svhn, random_split_train_val


def softmax(predictions):
    m = np.max(predictions, axis=1)
    m = m[:, np.newaxis]
    exps = np.exp(predictions - m)
    div = np.sum(exps, axis=1)
    div = div[:, np.newaxis]
    return exps / div


def cross_entropy_loss(probs, target_index):
    return -np.log(probs[np.arange(probs.shape[0]), target_index.reshape(-1, target_index.shape[0])]).sum()


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs
    dprediction[np.arange(dprediction.shape[0]), target_index.reshape(-1, target_index.shape[0])] -= 1
    return loss, dprediction


class FCLayer:
    def __init__(self, n1, n2):
        self.w = np.random.randn(n1, n2) / 1000
        self._saved = None

    def forward(self, X):
        result = np.dot(X, self.w)
        self._saved = X
        return result

    def backward(self, d_top):
        d_input = self._saved.T
        return np.dot(d_input, d_top)


class ReluLayer:
    def __init__(self):
        self._saved = None

    def forward(self, X):
        self._saved = X
        return np.where(X > 0, X, 0)

    def backward(self, d_top):
        d_input = np.where(self._saved > 0, 1, 0)
        return d_input * d_top


def f_relu(x):
    value = np.where(x > 0, x, 0)
    grad = np.where(x > 0, 1, 0)
    return value, grad


def check_gradient(f, x, delta=1e-5, tol=1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''

    assert isinstance(x, np.ndarray)
    assert x.dtype == float

    orig_x = x.copy()
    fx, analytic_grad = f(x)
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]
        h = np.zeros(x.shape)
        h[ix] = delta
        fxh, _ = f(x + h)
        numeric_grad_at_ix = (fxh - fx) / delta

        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (
            ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True


def prepare_for_linear_classifier(train_X, test_X):
    train_flat = train_X.reshape(train_X.shape[0], -1).astype(float) / 255.0
    test_flat = test_X.reshape(test_X.shape[0], -1).astype(float) / 255.0

    # Subtract mean
    mean_image = np.mean(train_flat, axis=0)
    train_flat -= mean_image
    test_flat -= mean_image

    # Add another channel with ones as a bias term
    train_flat_with_ones = np.hstack([train_flat, np.ones((train_X.shape[0], 1))])
    test_flat_with_ones = np.hstack([test_flat, np.ones((test_X.shape[0], 1))])
    return train_flat_with_ones, test_flat_with_ones


if __name__ == '__main__':
    # np.random.seed(1)
    # # points (-1,-1) (1, 1)
    # n = 64
    # x1 = np.reshape(-1 + np.random.randn(n), (-1, 1))
    # y1 = np.reshape(-1 + np.random.randn(n), (-1, 1))
    # x2 = np.reshape(1 + np.random.randn(n), (-1, 1))
    # y2 = np.reshape(1 + np.random.randn(n), (-1, 1))
    #
    # xy1 = np.concatenate([x1, y1], axis=1)
    # xy2 = np.concatenate([x2, y2], axis=1)
    #
    # x = np.concatenate([xy1, xy2])
    # expected_labels = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    #
    # # check_gradient(f_relu, np.array([1.0]))
    #

    train_X, train_y, test_X, test_y = load_svhn("../Resources", max_train=10000, max_test=1000)
    train_X, test_X = prepare_for_linear_classifier(train_X, test_X)
    train_X, train_y, val_X, val_y = random_split_train_val(train_X, train_y, num_val=1000)

    batch_size = 100
    n_features = train_X.shape[1]
    num_classes = np.max(train_y) + 1

    fc1 = FCLayer(n_features, num_classes)
    # relu = ReluLayer()

    loss_history = []
    for epoch in tqdm(range(10)):
        i = np.arange(train_X.shape[0])
        np.random.shuffle(i)
        sections = np.arange(batch_size, train_X.shape[0], batch_size)
        batches_indices = np.array_split(i, sections)

        for batch_idx in batches_indices:
            batchX = train_X[batch_idx]
            batchY = train_y[batch_idx]

            z = fc1.forward(batchX)
            m = softmax(z)
            loss, d_pred = softmax_with_cross_entropy(m, batchY)

            d_fc1 = fc1.backward(1)
            lr = 0.001
            fc1.w -= lr * d_fc1

            # loss, grad = linear_softmax(batchX, self.W, batchY)
            # l2_loss, l2_grad = l2_regularization(self.W, reg)
            # loss += l2_loss
            # grad += l2_grad
            # self.W -= learning_rate * grad
        print("Epoch %i, loss: %f" % (epoch, loss))
        loss_history.append(loss)



        # z = fc1.forward(x)
        # a = relu.forward(z)
        # m = softmax(a)
        # loss, d_pred = softmax_with_cross_entropy(m, expected_labels)
        #
        #
        # # print("--------")
        # d_L_relu = relu.backward(d_top=d_pred)
        # # print(d_relu)
        #
        # d_fc1 = fc1.backward(d_L_relu)
        # lr = 0.1
        # fc1.w -= lr * d_fc1

    # print(np.sum(np.argmax(m, axis=1) == expected_labels) / (2 * n))


    print(1)
