import numpy as np

import backend
import nn


class Model(object):
    """Base model class for the different applications"""

    def __init__(self):
        self.get_data_and_monitor = None
        self.learning_rate = 0.0

    def run(self, x, y=None):
        raise NotImplementedError("Model.run must be overriden by subclasses")

    def train(self):
        """
        Train the model.

        `get_data_and_monitor` will yield data points one at a time. In between
        yielding data points, it will also monitor performance, draw graphics,
        and assist with automated grading. The model (self) is passed as an
        argument to `get_data_and_monitor`, which allows the monitoring code to
        evaluate the model on examples from the validation set.
        """
        for x, y in self.get_data_and_monitor(self):
            graph = self.run(x, y)
            graph.backprop()
            graph.step(self.learning_rate)


class RegressionModel(Model):
    """
    TODO: Question 4 - [Application] Regression

    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        self.learning_rate = .05
        self.weight1 = nn.Variable(1, 200)
        self.weight2 = nn.Variable(200, 1)
        self.bias1 = nn.Variable(200)
        self.bias2 = nn.Variable(1)
        "*** YOUR CODE HERE ***"

    def run(self, x, y=None):
        """
        TODO: Question 4 - [Application] Regression

        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            graph = nn.Graph([self.weight1, self.weight2, self.bias1, self.bias2])
            input_x = nn.Input(graph, x)
            input_y = nn.Input(graph, y)
            xw1 = nn.MatrixMultiply(graph, input_x, self.weight1)
            xw1_plus_b1 = nn.MatrixVectorAdd(graph, xw1, self.bias1)
            xw1_ReLUd = nn.ReLU(graph, xw1_plus_b1)
            xw2 = nn.MatrixMultiply(graph, xw1_ReLUd, self.weight2)
            xw2_plus_b2 = nn.MatrixVectorAdd(graph, xw2, self.bias2)
            loss = nn.SquareLoss(graph, xw2_plus_b2, input_y)
            return graph
            "*** YOUR CODE HERE ***"
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            graph = nn.Graph([self.weight1, self.weight2, self.bias1, self.bias2])
            input_x = nn.Input(graph, x)
            xw1 = nn.MatrixMultiply(graph, input_x, self.weight1)
            xw1_plus_b1 = nn.MatrixVectorAdd(graph, xw1, self.bias1)
            xw1_ReLUd = nn.ReLU(graph, xw1_plus_b1)
            xw2 = nn.MatrixMultiply(graph, xw1_ReLUd, self.weight2)
            xw2_plus_b2 = nn.MatrixVectorAdd(graph, xw2, self.bias2)
            return graph.get_output(xw2_plus_b2)


class OddRegressionModel(Model):
    """
    TODO: Question 5 - [Application] OddRegression

    A neural network model for approximating a function that maps from real
    numbers to real numbers.

    Unlike RegressionModel, the OddRegressionModel must be structurally
    constrained to represent an odd function, i.e. it must always satisfy the
    property f(x) = -f(-x) at all points during training.
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = .02
        self.weight1 = nn.Variable(1, 50)
        self.weight2 = nn.Variable(50, 1)
        self.bias1 = nn.Variable(50)
        self.bias2 = nn.Variable(1)

    def run(self, x, y=None):
        """
        TODO: Question 5 - [Application] OddRegression

        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            graph = nn.Graph([self.weight1, self.weight2, self.bias1, self.bias2])
            pos_x = nn.Input(graph, x)
            input_y = nn.Input(graph, y)
            negone = nn.Input(graph, np.array([[-1.0]]))
            neg_x = nn.MatrixMultiply(graph, pos_x, negone)

            xw1_pos = nn.MatrixMultiply(graph, pos_x, self.weight1)
            xw1_plus_b1_pos = nn.MatrixVectorAdd(graph, xw1_pos, self.bias1)
            xw1_ReLUd_pos = nn.ReLU(graph, xw1_plus_b1_pos)
            xw2_pos = nn.MatrixMultiply(graph, xw1_ReLUd_pos, self.weight2)
            xw2_plus_b2_pos = nn.MatrixVectorAdd(graph, xw2_pos, self.bias2)

            xw1_neg = nn.MatrixMultiply(graph, neg_x, self.weight1)
            xw1_plus_b1_neg = nn.MatrixVectorAdd(graph, xw1_neg, self.bias1)
            xw1_ReLUd_neg = nn.ReLU(graph, xw1_plus_b1_neg)
            xw2_neg = nn.MatrixMultiply(graph, xw1_ReLUd_neg, self.weight2)
            xw2_plus_b2_neg = nn.MatrixVectorAdd(graph, xw2_neg, self.bias2)
            xw2_negation = nn.MatrixMultiply(graph, xw2_plus_b2_neg, negone)

            pos_plus_neg = nn.Add(graph, xw2_plus_b2_pos, xw2_negation)

            loss_pos = nn.SquareLoss(graph, pos_plus_neg, input_y)
            return graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            graph = nn.Graph([self.weight1, self.weight2, self.bias1, self.bias2])
            pos_x = nn.Input(graph, x)
            negone = nn.Input(graph, np.array([[-1.0]]))
            neg_x = nn.MatrixMultiply(graph, pos_x, negone)

            xw1_pos = nn.MatrixMultiply(graph, pos_x, self.weight1)
            xw1_plus_b1_pos = nn.MatrixVectorAdd(graph, xw1_pos, self.bias1)
            xw1_ReLUd_pos = nn.ReLU(graph, xw1_plus_b1_pos)
            xw2_pos = nn.MatrixMultiply(graph, xw1_ReLUd_pos, self.weight2)
            xw2_plus_b2_pos = nn.MatrixVectorAdd(graph, xw2_pos, self.bias2)

            xw1_neg = nn.MatrixMultiply(graph, neg_x, self.weight1)
            xw1_plus_b1_neg = nn.MatrixVectorAdd(graph, xw1_neg, self.bias1)
            xw1_ReLUd_neg = nn.ReLU(graph, xw1_plus_b1_neg)
            xw2_neg = nn.MatrixMultiply(graph, xw1_ReLUd_neg, self.weight2)
            xw2_plus_b2_neg = nn.MatrixVectorAdd(graph, xw2_neg, self.bias2)
            xw2_negation = nn.MatrixMultiply(graph, xw2_plus_b2_neg, negone)

            pos_plus_neg = nn.Add(graph, xw2_plus_b2_pos, xw2_negation)
            return graph.get_output(pos_plus_neg)


class DigitClassificationModel(Model):
    """
    TODO: Question 6 - [Application] Digit Classification

    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_digit_classification

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"

        self.learning_rate = .6
        self.weight1 = nn.Variable(784, 128)
        self.weight2 = nn.Variable(128, 64)
        self.weight3 = nn.Variable(64, 10)
        self.bias1 = nn.Variable(128)
        self.bias2 = nn.Variable(64)
        self.bias3 = nn.Variable(10)

    def run(self, x, y=None):
        """
        TODO: Question 6 - [Application] Digit Classification

        Runs the model for a batch of examples.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 10) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should predict a (batch_size x 10) numpy array of scores,
        where higher scores correspond to greater probability of the image
        belonging to a particular class. You should use `nn.SoftmaxLoss` as your
        training loss.

        Inputs:
            x: a (batch_size x 784) numpy array
            y: a (batch_size x 10) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 10) numpy array of scores (aka logits)
        """
        "*** YOUR CODE HERE ***"

        if y is not None:
            "*** YOUR CODE HERE ***"

            graph = nn.Graph([self.weight1, self.weight2, self.weight3, self.bias1, self.bias2, self.bias3])
            input_x = nn.Input(graph, x)
            input_y = nn.Input(graph, y)
            xw1 = nn.MatrixMultiply(graph, input_x, self.weight1)
            xw1_plus_b1 = nn.MatrixVectorAdd(graph, xw1, self.bias1)
            xw1_ReLUd = nn.ReLU(graph, xw1_plus_b1)
            xw2 = nn.MatrixMultiply(graph, xw1_ReLUd, self.weight2)
            xw2_plus_b2 = nn.MatrixVectorAdd(graph, xw2, self.bias2)
            xw2_ReLUd = nn.ReLU(graph, xw2_plus_b2)
            xw3 = nn.MatrixMultiply(graph, xw2_ReLUd, self.weight3)
            xw3_plus_b3 = nn.MatrixVectorAdd(graph, xw3, self.bias3)
            loss = nn.SoftmaxLoss(graph, xw3_plus_b3, input_y)
            return graph
        else:
            "*** YOUR CODE HERE ***"

            graph = nn.Graph([self.weight1, self.weight2, self.weight3, self.bias3, self.bias1, self.bias2])
            input_x = nn.Input(graph, x)
            xw1 = nn.MatrixMultiply(graph, input_x, self.weight1)
            xw1_plus_b1 = nn.MatrixVectorAdd(graph, xw1, self.bias1)
            xw1_ReLUd = nn.ReLU(graph, xw1_plus_b1)
            xw2 = nn.MatrixMultiply(graph, xw1_ReLUd, self.weight2)
            xw2_plus_b2 = nn.MatrixVectorAdd(graph, xw2, self.bias2)
            xw2_ReLUd = nn.ReLU(graph, xw2_plus_b2)
            xw3 = nn.MatrixMultiply(graph, xw2_ReLUd, self.weight3)
            xw3_plus_b3 = nn.MatrixVectorAdd(graph, xw3, self.bias3)
            return graph.get_output(xw3_plus_b3)


class DeepQModel(Model):
    """
    TODO: Question 7 - [Application] Reinforcement Learning

    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.

    (We recommend that you implement the RegressionModel before working on this
    part of the project.)
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_rl

        self.num_actions = 2
        self.state_size = 4

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = .01

        self.weight1 = nn.Variable(4, 100)
        self.weight2 = nn.Variable(100, 2)
        self.bias1 = nn.Variable(100)
        self.bias2 = nn.Variable(2)

    def run(self, states, Q_target=None):
        """
        TODO: Question 7 - [Application] Reinforcement Learning

        Runs the DQN for a batch of states.

        The DQN takes the state and computes Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]

        When Q_target == None, return the matrix of Q-values currently computed
        by the network for the input states.

        When Q_target is passed, it will contain the Q-values which the network
        should be producing for the current states. You must return a nn.Graph
        which computes the training loss between your current Q-value
        predictions and these target values, using nn.SquareLoss.

        Inputs:
            states: a (batch_size x 4) numpy array
            Q_target: a (batch_size x 2) numpy array, or None
        Output:
            (if Q_target is not None) A nn.Graph instance, where the last added
                node is the loss
            (if Q_target is None) A (batch_size x 2) numpy array of Q-value
                scores, for the two actions
        """
        "*** YOUR CODE HERE ***"

        if Q_target is not None:

            graph = nn.Graph([self.weight1, self.weight2, self.bias1, self.bias2])
            input_x = nn.Input(graph, states)
            input_y = nn.Input(graph, Q_target)
            xw1 = nn.MatrixMultiply(graph, input_x, self.weight1)
            xw1_plus_b1 = nn.MatrixVectorAdd(graph, xw1, self.bias1)
            xw1_ReLUd = nn.ReLU(graph, xw1_plus_b1)
            xw2 = nn.MatrixMultiply(graph, xw1_ReLUd, self.weight2)
            xw2_plus_b2 = nn.MatrixVectorAdd(graph, xw2, self.bias2)
            loss = nn.SquareLoss(graph, xw2_plus_b2, input_y)
            return graph
            "*** YOUR CODE HERE ***"
        else:
            graph = nn.Graph([self.weight1, self.weight2, self.bias1, self.bias2])
            input_x = nn.Input(graph, states)
            xw1 = nn.MatrixMultiply(graph, input_x, self.weight1)
            xw1_plus_b1 = nn.MatrixVectorAdd(graph, xw1, self.bias1)
            xw1_ReLUd = nn.ReLU(graph, xw1_plus_b1)
            xw2 = nn.MatrixMultiply(graph, xw1_ReLUd, self.weight2)
            xw2_plus_b2 = nn.MatrixVectorAdd(graph, xw2, self.bias2)
            return graph.get_output(xw2_plus_b2)
            "*** YOUR CODE HERE ***"

    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.

        Inputs:
            state: a (1 x 4) numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions)
        else:
            scores = self.run(state)
            return int(np.argmax(scores))


class LanguageIDModel(Model):
    """
    TODO: Question 8 - [Application] Language Identification

    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_lang_id

        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = .025

        self.weight1 = nn.Variable(47, 47)
        self.weight2 = nn.Variable(250, 75)
        self.weight3 = nn.Variable(47, 5)
        self.weight4 = nn.Variable(5, len(self.languages))
        self.bias1 = nn.Variable(32)
        self.bias2 = nn.Variable(5)
        self.bias3 = nn.Variable(len(self.languages))

    def run(self, xs, y=None):
        """
        TODO: Question 8 - [Application] Language Identification

        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        (batch_size x self.num_chars) numpy array, where every row in the array
        is a one-hot vector encoding of a character. For example, if we have a
        batch of 8 three-letter words where the last word is "cat", we will have
        xs[1][7,0] == 1. Here the index 0 reflects the fact that the letter "a"
        is the inital (0th) letter of our combined alphabet for this task.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 5) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node that represents a (batch_size x hidden_size)
        array, for your choice of hidden_size. It should then calculate a
        (batch_size x 5) numpy array of scores, where higher scores correspond
        to greater probability of the word originating from a particular
        language. You should use `nn.SoftmaxLoss` as your training loss.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a (batch_size x self.num_chars) numpy array
            y: a (batch_size x 5) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 5) numpy array of scores (aka logits)

        Hint: you may use the batch_size variable in your code
        """
        batch_size = xs[0].shape[0]

        "*** YOUR CODE HERE ***"

        if y is not None:
            #var_h = nn.Variable(batch_size, 16)
            graph = nn.Graph([self.weight1, self.weight4, self.bias1, self.bias2, self.weight2, self.weight3, self.bias3])
            var_h = nn.Input(graph, np.zeros((batch_size, 5)))
            input_y = nn.Input(graph, y)
            for char in xs:
                input_char = nn.Input(graph, char)
                xw1 = nn.MatrixMultiply(graph, input_char, self.weight1)
                #xw1_plus_b1 = nn.MatrixVectorAdd(graph, xw1, self.bias1)
                #xw2 = nn.MatrixMultiply(graph, xw1, self.weight2)
                #xw2_plus_b2 = nn.MatrixVectorAdd(graph, xw2, self.bias2)
                #xw2_ReLUd = nn.ReLU(graph, xw2)
                xw3 = nn.MatrixMultiply(graph, xw1, self.weight3)
                xw2_plus_b2 = nn.MatrixVectorAdd(graph, xw3, self.bias2)
                xw3_ReLUd = nn.ReLU(graph, xw2_plus_b2)
                var_h = nn.Add(graph, var_h, xw3_ReLUd)
            xw4 = nn.MatrixMultiply(graph, var_h, self.weight4)
            xw4_plus_b3 = nn.MatrixVectorAdd(graph, xw4, self.bias3)
            #xw4_ReLUd = nn.ReLU(graph, xw4_plus_b3)
            loss = nn.SoftmaxLoss(graph, xw4_plus_b3, input_y)
            return graph
            "*** YOUR CODE HERE ***"
        else:
            #var_h = nn.Variable(batch_size, 16)
            graph = nn.Graph([self.weight1, self.weight4, self.bias1, self.weight2, self.bias2, self.weight3, self.bias3])
            var_h = nn.Input(graph, np.zeros((batch_size, 5)))
            for char in xs:
                input_char = nn.Input(graph, char)
                #print graph.get_output(input_char).shape
                #return graph.get_output(input_char)
                xw1 = nn.MatrixMultiply(graph, input_char, self.weight1)
                #xw1_plus_b1 = nn.MatrixVectorAdd(graph, xw1, self.bias1)
                #xw2 = nn.MatrixMultiply(graph, xw1, self.weight2)
                #xw2_plus_b2 = nn.MatrixVectorAdd(graph, xw2, self.bias2)
                #xw2_ReLUd = nn.ReLU(graph, xw2)
                xw3 = nn.MatrixMultiply(graph, xw1, self.weight3)
                xw2_plus_b2 = nn.MatrixVectorAdd(graph, xw3, self.bias2)
                xw3_ReLUd = nn.ReLU(graph, xw2_plus_b2)
                var_h = nn.Add(graph, var_h, xw3_ReLUd)
            xw4 = nn.MatrixMultiply(graph, var_h, self.weight4)
            xw4_plus_b3 = nn.MatrixVectorAdd(graph, xw4, self.bias3)
            #xw4_ReLUd = nn.ReLU(graph, xw4_plus_b3)
            return  graph.get_output(xw4_plus_b3)
            "*** YOUR CODE HERE ***"
