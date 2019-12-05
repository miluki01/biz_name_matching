from transformers import BertModel, BertTokenizer, BertConfig
from model import BertForCopyRightNameMatching
from dataloader import TitleDataset
from torch.utils import data
import torch.optim as optim
import torch


params = {
    'batch_size': 8,
    'shuffle': True,
    'num_workers': 8,
    'drop_last': True
}

opim_params = {
    "lr": 0.00000001
}

model_params = {
    # "max_full_text_len": 60,
    "batch_size": params['batch_size']
}

accumulation_steps = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
    # print(config)

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = BertForCopyRightNameMatching(config=config)
    training_set = TitleDataset('data/train.csv',
                                tokenizer=bert_tokenizer,
                                max_full_text_len=512)
    test_set = TitleDataset('data/test.csv',
                            tokenizer=bert_tokenizer,
                            max_full_text_len=512)

    training_generator = data.DataLoader(training_set, **params)
    test_generator = data.DataLoader(test_set, **params)

    return (training_generator, test_generator, model)


def train(training_generator, test_generator, model, device='cpu', max_epochs=2):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), **opim_params)
    for epoch in range(max_epochs):
        running_loss = 0.0
        model.zero_grad()
        for i, (input_ids, label) in enumerate(training_generator, 0):
            input_ids, label = input_ids.to(device), label.to(device)
            # print(input_ids.size())
            # print(label.size())
            outputs = model(input_ids=input_ids, labels=label)
            optimizer.zero_grad()

            loss = outputs[0].squeeze()
            loss.backward()


            if (i+1) % accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:    # print every 10 mini-batches
                print('Epoch: {}, step: {}, loss: {}'.format(epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    running_loss = 0.0
    with torch.set_grad_enabled(False):
        for input_ids, label in test_generator:
            outputs = model(input_ids, labels=label)

            loss = outputs[0].squeeze()
            running_loss += loss.item()
    print("Validation loss: {}".format(running_loss))

if __name__ == "__main__":
    training_generator, test_generator, model = main()
    # input_ids, labels_1, labels_2 = next(iter(training_generator))
    train(training_generator=training_generator,
            test_generator=test_generator,
            model=model, device=device,
            max_epochs=10)
