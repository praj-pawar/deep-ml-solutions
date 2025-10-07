import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        """
        Processes a sequence of inputs and returns:
        - all hidden states
        - final hidden state
        - final cell state
        """
        h_t = initial_hidden_state
        c_t = initial_cell_state
        hidden_states = []

        for t in range(len(x)):
            x_t = x[t].reshape(-1, 1)  # (input_size, 1)
            combined = np.vstack((h_t, x_t))  # (hidden_size + input_size, 1)

            f_t = self.sigmoid(self.Wf @ combined + self.bf)
            i_t = self.sigmoid(self.Wi @ combined + self.bi)
            c_hat_t = np.tanh(self.Wc @ combined + self.bc)
            c_t = f_t * c_t + i_t * c_hat_t
            o_t = self.sigmoid(self.Wo @ combined + self.bo)
            h_t = o_t * np.tanh(c_t)

            hidden_states.append(h_t)

        hidden_states = np.hstack(hidden_states)  # concatenate across time steps
        return hidden_states, h_t, c_t
