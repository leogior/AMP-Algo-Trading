from torch import nn
import torch



class LSTMDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        sequence_length=None,
        predict_backward=True,
        num_layers=1,
    ):
        super().__init__()

        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.predict_backward = predict_backward
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.lstm = (
            None
            if num_layers <= 1
            else nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers - 1,
            )
        )
        self.linear = (
            None
            if input_size == hidden_size
            else nn.Linear(hidden_size, input_size)
        )

    def forward(self, h, sequence_length=None):
        """Computes the forward pass.

        Parameters
        ----------
        x:
            Input of shape (batch_size, input_size)

        Returns
        -------
        Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            Decoder outputs (output, (h, c)) where output has the shape (sequence_length, batch_size, input_size).
        """

        if sequence_length is None:
            sequence_length = self.sequence_length
        x_hat = torch.empty(sequence_length, h.shape[0], self.hidden_size)
        
        for t in range(sequence_length):
            if t == 0:
                h, c = self.cell(h)
            else:
                input = h if self.linear is None else self.linear(h)
                h, c = self.cell(input, (h, c))
            t_predicted = -t if self.predict_backward else t
            x_hat[t_predicted] = h

        if self.lstm is not None:
            x_hat = self.lstm(x_hat)

        return x_hat, (h, c)

class LstmModule(nn.Module):
    def __init__(self, n_features, hidden_size=32, n_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
        )
        self.decoder = LSTMDecoder(
            input_size=hidden_size,
            hidden_size=n_features,
            predict_backward=True,
        )
        self.fc = nn.Linear(hidden_size, 1)  # maps final encoder state to scalar output

    def forward(self, X, **kwargs):
        # print(f"Input shape: {X.shape}")
        
        if X.dim() == 2:
            X = X.unsqueeze(1)  # adds batch dimension: (seq_len, 1, n_features)

        encoder_output, (hn, cn) = self.encoder(X)
        final_output = encoder_output[-1, 0, :]  # last timestep, first batch
        out = self.fc(final_output)
        return out.squeeze()