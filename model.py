from torch import nn



class MyModel(nn.Module):
    def __init__(self, bert, config):
        super().__init__()
        self.model = bert
        self.num_labels = 3
        self.project = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, text, labels=None):
        x = self.model(text['input_ids'], text['attention_mask']).last_hidden_state

        logits = self.project(x)

        loss_fct = nn.BCEWithLogitsLoss()

        if labels is not None:
            loss = loss_fct(logits, labels)
            return logits, loss
        return logits
