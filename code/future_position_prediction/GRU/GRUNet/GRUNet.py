class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, output_cor_dim, t_len,bias=True):
        super(GRUNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bias = bias
        self.dropout = [0, 0]
        self.output_cor_dim = output_cor_dim

        # self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.gru = customGRUCell(input_dim, hidden_dim )
        self.fc = nn.Linear(hidden_dim, hidden_dim)

        # self.bounding_box = bboxPredictor(hidden_dim, 4*t_len, dropout=[0, 0])
        self.bounding_box = MidPred(hidden_dim, 4*t_len, dropout=[0, 0])

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
        # self.dense1 = torch.nn.Linear(hidden_dim+output_cor_dim, 128)
        # self.dense2 = torch.nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h, bbox, dist_out):
        x = torch.squeeze(x,0)
        h = torch.squeeze(h,0)
        bbox = torch.squeeze(bbox,0)
        # dist = torch.squeeze(dist,0)

        if h is None:
            if torch.cuda.is_available():
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size).cuda())
            else:
                h0 = Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))

        else:
             h0 = h

        h = self.gru(x, h, bbox)  # out: confidence of a object is present
        out = self.fc(h)
        h = h.unsqueeze(0)
        out= out.unsqueeze(0)
        out = torch.cat([out, dist_out], dim=-1) ###temp comment: distance concatenated
        f_box = self.bounding_box(out)  # dim: 40, bbox (x,y, w, h)
        out = F.dropout(out[:, -1], self.dropout[0])  # optional
        return out, h, f_box
