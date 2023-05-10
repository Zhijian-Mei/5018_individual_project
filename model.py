from torch import nn


class MyModel(nn.Module):
    def __init__(self, bert, config, length=512):
        super().__init__()
        self.model = bert
        self.num_labels = 3
        self.fc = nn.Linear(config.hidden_size, self.num_labels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, text, labels=None):
        cls_hs = self.model(text['input_ids'], text['attention_mask']).pooler_output
        x = self.dropout(cls_hs)
        logits = self.fc(x)

        loss_fct = nn.CrossEntropyLoss()

        if labels is not None:
            loss = loss_fct(logits, labels)
            return logits, loss
        return logits
