import numpy as np
import utils
import typing
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    X = X.astype(np.float64)
    X -= np.mean(X)
    X /= X.std()

    bias_trick = np.zeros(shape=(X.shape[0], 1)) + 1
    X = np.append(X, bias_trick, axis=1)
    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """

    cross_entropies = -targets * np.log(outputs)
    cross_entropy_error = np.sum(cross_entropies) / targets.shape[0]
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    # TODO: Implement this function (copy from last assignment)
    return cross_entropy_error


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            w = np.zeros(w_shape)
            self.ws.append(w)
            prev = size

        fan_in = [self.I]
        for i in neurons_per_layer:
            fan_in.append(i)

        if use_improved_weight_init:
            for i in range(len(fan_in) - 1):
                self.ws[i] = np.random.normal(
                    0, 1 / np.sqrt(fan_in[i+1]), (fan_in[i], fan_in[i+1]))

            # Implementation for one hidden layer
            """
            fan_in = [self.I, neurons_per_layer[0]]
            self.ws[0] = np.random.normal(
                0, 1/np.sqrt(fan_in[0]), (self.I, neurons_per_layer[0]))

            self.ws[1] = np.random.normal(
                0, 1/np.sqrt(fan_in[1]), (neurons_per_layer[0], neurons_per_layer[1]))
            """
        else:
            for i in range(len(fan_in) - 1):
                self.ws[i] = np.random.uniform(
                    -1, 1, (fan_in[i], fan_in[i+1]))

        self.grads = [None for i in range(len(self.ws))]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For peforming the backward pass, you can save intermediate activations in varialbes in the forward pass.
        # such as self.hidden_layer_ouput = ...

        # Hidden layer outputs
        self.hidden_layer_output = []
        if self.use_improved_sigmoid:
            self.hidden_layer_output.append(
                1.7159*np.tanh(2*X.dot(self.ws[0])/3))
            for i in range(len(self.neurons_per_layer) - 2):
                self.hidden_layer_output.append(1.7159 *
                                                np.tanh(2*self.hidden_layer_output[i].dot(self.ws[i+1])/3))
        else:
            self.hidden_layer_output.append(
                1 / (1 + np.exp(-X.dot(self.ws[0]))))
            for i in range(len(self.neurons_per_layer) - 2):
                self.hidden_layer_output.append(
                    1 / (1 + np.exp(-self.hidden_layer_output[i].dot(self.ws[i+1]))))

        """
        Implementation for one HL:
        if self.use_improved_sigmoid:
            self.hidden_layer_output = 1.7159*np.tanh(2*X.dot(self.ws[0])/3)
        else:
            self.hidden_layer_output = 1 / (1 + np.exp(-X.dot(self.ws[0])))
        """

        # Model output
        ez = np.exp(self.hidden_layer_output[-1].dot(self.ws[-1]))
        ez_sum = ez.sum(axis=1, keepdims=True)
        y = np.divide(ez, ez_sum)

        return y

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        # TODO: Denne funksjonen kan nok implementeres med np-arrays i stedet for lister
        #       og med kun 1 for-loop
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"

        # Compute sigmoid derivatives
        sigmoid_dot = []
        if self.use_improved_sigmoid:
            z = X.dot(self.ws[0])
            sigmoid_dot.append(
                (1.7159 * 2) / (3*(np.power(np.cosh(2*z / 3), 2))))
            for i in range(len(self.neurons_per_layer) - 2):
                z = self.hidden_layer_output[i].dot(self.ws[i+1])
                sigmoid_dot.append(
                    (1.7159 * 2) / (3*(np.power(np.cosh(2*z / 3), 2))))
        else:
            for i in range(len(self.neurons_per_layer) - 1):
                sigmoid_dot.append(self.hidden_layer_output[i] *
                                   (1 - self.hidden_layer_output[i]))

        # Implementation for one HL:
        """
        if self.use_improved_sigmoid:
            sigmoid_dot = 2.28787/(np.cosh(2*self.hidden_layer_output[0])+1)
        else:
            sigmoid_dot = self.hidden_layer_output *
                (1 - self.hidden_layer_output[0])
        """

        # Compute delta_k
        delta_k = outputs - targets

        # Compute delta_j's
        delta_j = []

        # delta_{L-1}
        delta_j.append(sigmoid_dot[-1] *
                       delta_k.dot(np.transpose(self.ws[-1])))

        for i in range(1, len(self.neurons_per_layer) - 1):
            delta_j.append(
                sigmoid_dot[-(i+1)]*delta_j[i-1].dot(np.transpose(self.ws[-(i+1)])))
        delta_j.reverse()

        # Compute gradients

        # Compute gradient for last->output layer
        self.grads[-1] = np.transpose(self.hidden_layer_output[-1]).dot(
            delta_k) / X.shape[0]

        # Gradient for input->first_hidden layer
        self.grads[0] = np.transpose(X).dot(
            delta_j[0]) / X.shape[0]

        # Gradients for hidden layers
        for i in range(1, len(self.neurons_per_layer)-1):
            self.grads[i] = np.transpose(
                self.hidden_layer_output[i-1]).dot(delta_j[i])/X.shape[0]

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # TODO: Implement this function (copy from last assignment)

    out = np.zeros(shape=(Y.shape[0], num_classes))
    for i in range(Y.shape[0]):
        out[i][Y[i]] = 1
    Y = out
    return Y


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited.
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 32, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
