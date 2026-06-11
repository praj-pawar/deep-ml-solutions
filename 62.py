import numpy as np


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the RNN with small random weights and zero biases.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x):
        """
        Forward pass over a sequence x of shape (T, input_size).
        Returns outputs stacked as (T, output_size, 1) and caches states for BPTT.
        """
        T = len(x)
        self.hidden_states = {-1: np.zeros((self.hidden_size, 1))}
        self.outputs = []

        for t in range(T):
            x_t = x[t].reshape(-1, 1)
            h_t = np.tanh(
                self.W_xh @ x_t
                + self.W_hh @ self.hidden_states[t - 1]
                + self.b_h
            )
            self.hidden_states[t] = h_t
            y_t = self.W_hy @ h_t + self.b_y
            self.outputs.append(y_t)

        # Return the raw stack -> shape (T, output_size, 1). No reshape/flatten.
        return np.array(self.outputs)

    def backward(self, x, y, learning_rate):
        """
        Backpropagation Through Time.
        Loss = sum_t [ 1/2 * MSE(y_pred_t, y_t) ], summed across time steps.
        Relies on the cache from the most recent forward(x) call.
        """
        T = len(x)

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(T)):
            x_t = x[t].reshape(-1, 1)
            target_t = y[t].reshape(-1, 1)
            y_t = self.outputs[t]
            h_t = self.hidden_states[t]
            h_prev = self.hidden_states[t - 1]

            # d(loss_t)/d(y_t); loss is 1/2 * summed squared error, so the
            # 1/2 cancels the 2 and there is no averaging over the output dim
            dy = y_t - target_t

            dW_hy += dy @ h_t.T
            db_y += dy

            dh = self.W_hy.T @ dy + dh_next
            dh_raw = (1 - h_t * h_t) * dh   # tanh'(z) = 1 - tanh(z)^2

            db_h += dh_raw
            dW_xh += dh_raw @ x_t.T
            dW_hh += dh_raw @ h_prev.T

            dh_next = self.W_hh.T @ dh_raw

        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y
