from torch import nn



class MyModel(nn.Module):
    def __init__(self, bert, config,length = 512):
        super().__init__()
        self.model = bert
        self.num_labels = 3
        self.project0 = nn.Linear(length,1)
        self.project1 = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, text, labels=None):
        x = self.model(text['input_ids'], text['attention_mask']).last_hidden_state
        x = x.reshape((x.shape[0],x.shape[2],x.shape[1]))
        x = self.project0(x).squeeze()
        logits = self.project1(x)

        loss_fct = nn.CrossEntropyLoss()

        if labels is not None:
            loss = loss_fct(logits, labels)
            return logits, loss
        return logits
