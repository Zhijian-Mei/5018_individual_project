import csv
import torch
from tqdm import trange
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score

# Define the query template and verbalizer functions
def query_template(premise, hypothesis):
    return f"NLI: premise: {premise} hypothesis: {hypothesis}"

def verbalizer(output):
    labels = ["entailment", "neutral", "contradiction"]
    label_id = torch.argmax(output).item()
    if label_id >= len(labels):
        return labels[-1] # return the default label if the index is out of range
    else:
        return labels[label_id]
device = torch.device(f'cuda:0')
# Load the pre-trained model and tokenizer
model_name = "t5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the training data from the CSV file
train_data_path = "data/validation_matched.csv"
train_premises = []
train_hypotheses = []
train_labels = []
test_premises = []
test_hypotheses = []
test_labels = []
with open(train_data_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    count = 0
    for row in reader:
        count += 1
        if count <2300:
          train_premises.append(row['premise'])
          train_hypotheses.append(row['hypothesis'])
                # Correct the label IDs to match their verbalizations
          if row['label'] == '2':
              train_labels.append("entailment")
          elif row['label'] == '1':
              train_labels.append("neutral")
          elif row['label'] == '0':
              train_labels.append("contradiction")
        else:
            test_premises.append(row['premise'])
            test_hypotheses.append(row['hypothesis'])
                # Correct the label IDs to match their verbalizations
            if row['label'] == '2':
                test_labels.append("entailment")
            elif row['label'] == '1':
                test_labels.append("neutral")
            elif row['label'] == '0':
                test_labels.append("contradiction")   
        if count > 2500:
            break         
        # train_labels.append(int(row['label']))

# Set up the training hyperparameters
batch_size = 4
learning_rate = 1e-4
num_epochs = 1

# Set up the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

# Train the model
model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}:")
    total_loss = 0.0
    for i in trange(0, len(train_premises)//2, batch_size):
        batch_premises = train_premises[i:i+batch_size]
        batch_hypotheses = train_hypotheses[i:i+batch_size]
        output = train_labels[i:i+batch_size]
        # batch_labels = torch.tensor(train_labels[i:i+batch_size], dtype=torch.long)
        optimizer.zero_grad()
        input_text = [query_template(premise, hypothesis) for premise, hypothesis in zip(batch_premises, batch_hypotheses)]

        input_ =  tokenizer.batch_encode_plus(
            input_text,
            max_length=256,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            ).to(device)
        output_ = tokenizer.batch_encode_plus(
            output,
            max_length=256,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            )
        labels = output_.input_ids.to(device)
        labels[labels==0] = -100
        # input_ids = tokenizer(input_text, padding=True, return_tensors='pt')
        output = model(input_.input_ids, attention_mask=input_.attention_mask, labels=labels)
        loss = output.loss
        print(loss.item())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Training loss: {total_loss:.3f}")

model.eval()

accuracy = 0.0
t = 0
# Train the model
for epoch in range(1):
    print(f"Epoch {epoch + 1}:")
    total_loss = 0.0
    for i in range(0, len(test_premises), batch_size):
        batch_premises = test_premises[i:i+batch_size]
        batch_hypotheses = test_hypotheses[i:i+batch_size]
        label = test_labels[i:i+batch_size]
        input_text = [query_template(premise, hypothesis) for premise, hypothesis in zip(batch_premises, batch_hypotheses)]
        # print(input_text)
        input_ =  tokenizer.batch_encode_plus(
            input_text,
            max_length=256,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            ).to(device)
        output = model.generate(input_ids=input_.input_ids,attention_mask=input_.attention_mask)
        output_ = tokenizer.batch_decode(output, skip_special_tokens=True)
        print(output_)
        print(label)
        k = 0
        count = len(output_)
        hit = 0
        for i in output_:
          if i == label[k]:
            hit += 1
          k += 1
        acc = hit/count
        accuracy += acc
        t += 1
print('final')
print(accuracy/t)

