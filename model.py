from torch import nn



class MyModel(nn.Module):
    def __init__(self, bert, config,length = 512):
        super().__init__()
        self.model = bert
        self.num_labels = 3
        self.fc1 = nn.Linear(config.hidden_size,512)
        self.fc2 = nn.Linear(512, self.num_labels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, text, labels=None):
        cls_hs = self.model(text['input_ids'], text['attention_mask']).pooler_output

        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)


        loss_fct  = nn.NLLLoss()

        if labels is not None:
            loss = loss_fct(x, labels)
            return x, loss
        return x
