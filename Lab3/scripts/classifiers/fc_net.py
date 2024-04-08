from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        for layer, dims in enumerate([(input_dim, hidden_dim), (hidden_dim, num_classes)], start=1):
            self.params[f'W{layer}'] = weight_scale * np.random.randn(*dims)  # Инициализация весов
            self.params[f'b{layer}'] = np.zeros(dims[-1])  # Инициализация смещений

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        # First layer: Affine - ReLU
        out1, cache1 = affine_relu_forward(X, W1, b1)
        # Second layer: Affine
        scores, cache2 = affine_forward(out1, W2, b2)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Compute the loss
        loss, dscores = softmax_loss(scores, y)
        # Add regularization
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        # Backward pass
        dout1, grads['W2'], grads['b2'] = affine_backward(dscores, cache2)
        dx, grads['W1'], grads['b1'] = affine_relu_backward(dout1, cache1)

        # Add regularization gradients
        grads['W2'] += self.reg * W2
        grads['W1'] += self.reg * W1

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        for i in range(self.num_layers):
            # Определение размерностей слоев
            layer_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            layer_output_dim = num_classes if i == self.num_layers - 1 else hidden_dims[i]

            # Ключи для параметров текущего слоя
            w_key, b_key = f"W{i + 1}", f"b{i + 1}"

            # Инициализация весов и смещений
            self.params[w_key] = np.random.randn(layer_input_dim, layer_output_dim) * weight_scale
            self.params[b_key] = np.zeros(layer_output_dim)

            # Проверка необходимости инициализации параметров нормализации
            if i < self.num_layers - 1 and self.normalization in {"batchnorm"}:
                gamma_key, beta_key = f"gamma{i + 1}", f"beta{i + 1}"
                self.params[gamma_key] = np.ones(layer_output_dim)
                self.params[beta_key] = np.zeros(layer_output_dim)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        caches = {}

        # Проходим по всем слоям, кроме последнего
        for layer_index in range(1, self.num_layers):
            # Строим ключи для весов и смещений текущего слоя
            weight_key = f"W{layer_index}"
            bias_key = f"b{layer_index}"

            # Инициализируем входной тензор для первого слоя
            if layer_index == 1:
                layer_input = X

            # Если используется нормализация по батчам
            if self.normalization == "batchnorm":
                # Прямой проход через полносвязный слой и batchnorm
                fc_output, fc_cache = affine_forward(layer_input, self.params[weight_key], self.params[bias_key])
                gamma_key = f"gamma{layer_index}"
                beta_key = f"beta{layer_index}"
                bn_output, bn_cache = batchnorm_forward(fc_output, self.params[gamma_key], self.params[beta_key],
                                                        self.bn_params[layer_index - 1])
                layer_output, relu_cache = relu_forward(bn_output)
                # Сохраняем кэш текущего слоя
                caches[layer_index] = (fc_cache, bn_cache, relu_cache)
            else:
                # Прямой проход через комбинированный слой affine + ReLU
                layer_output, cache = affine_relu_forward(layer_input, self.params[weight_key], self.params[bias_key])
                caches[layer_index] = cache

            # Если используется dropout
            if self.use_dropout:
                layer_output, dropout_cache = dropout_forward(layer_output, self.dropout_param)
                caches[f"dropout{layer_index}"] = dropout_cache

            # Подготовка входа для следующего слоя
            layer_input = layer_output

        # Проход через последний полносвязный слой
        final_weight_key = f"W{self.num_layers}"
        final_bias_key = f"b{self.num_layers}"
        scores, cache = affine_forward(layer_input, self.params[final_weight_key], self.params[final_bias_key])
        caches[self.num_layers] = cache

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dscores = softmax_loss(scores, y)

        # Итерация по слоям в обратном порядке для обратного распространения.
        for layer_idx in reversed(range(1, self.num_layers + 1)):
            W_key = f"W{layer_idx}"
            b_key = f"b{layer_idx}"
            W = self.params[W_key]

            # Добавляем регуляризационные потери к общей потере.
            loss += 0.5 * self.reg * np.sum(W ** 2)

            # Обработка последнего слоя отдельно
            if layer_idx == self.num_layers:
                dout, grads[W_key], grads[b_key] = affine_backward(dscores, caches[layer_idx])
            else:
                # Предварительная обработка для dropout, если он используется.
                if self.use_dropout:
                    dout = dropout_backward(dout, caches[f"dropout{layer_idx}"])

                if self.normalization == "batchnorm":
                    fc_cache, bn_cache, relu_cache = caches[layer_idx]
                    dout = relu_backward(dout, relu_cache)
                    dout, grads[f"gamma{layer_idx}"], grads[f"beta{layer_idx}"] = batchnorm_backward(dout, bn_cache)
                    dout, grads[W_key], grads[b_key] = affine_backward(dout, fc_cache)
                else:
                    dout, grads[W_key], grads[b_key] = affine_relu_backward(dout, caches[layer_idx])

            # Применяем регуляризацию к градиентам весов.
            grads[W_key] += self.reg * W

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
