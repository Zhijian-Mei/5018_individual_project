from torch import nn


class MyModel(nn.Module):
    def __init__(self, bert, config, length=256):
        super().__init__()
        self.model = bert
        self.num_labels = 3
        self.sequence_length = length
        self.fc1 = nn.Linear(config.hidden_size,1)
        self.fc2 = nn.Linear(self.sequence_length,self.num_labels)
        # self.fc = nn.Linear(config.hidden_size, self.num_labels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, text, labels=None):
        output = self.model(text['input_ids'], text['attention_mask'])
        last_hidden_state = output[0]
        cls_hs = output[1]
        x = self.fc1(last_hidden_state).squeeze()
        x = self.fc2(x)
        x = self.dropout(x)
        print(x.shape)
        quit()
        logits = self.fc(x)

        loss_fct = nn.CrossEntropyLoss()

        if labels is not None:
            loss = loss_fct(logits, labels)
            return logits, loss
        return logits
