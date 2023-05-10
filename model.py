from torch import nn


class MyModel(nn.Module):
    def __init__(self, bert, config, length=256):
        super().__init__()
        self.model = bert
        self.num_labels = 3
        self.sequence_length = length
        # self.fc1 = nn.Linear(config.hidden_size,1)
        # self.fc2 = nn.Linear(self.sequence_length,self.num_labels)
        self.fc = nn.Linear(config.hidden_size, self.num_labels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, text, labels=None):
        output = self.model(text['input_ids'], text['attention_mask'])
        cls_hs = output.last_hidden_state[:,0,:]
        print(cls_hs.shape)
        quit()
        logits = self.fc(self.dropout(cls_hs))

        loss_fct = nn.CrossEntropyLoss()

        if labels is not None:
            loss = loss_fct(logits, labels)
            return logits, loss
        return logits
