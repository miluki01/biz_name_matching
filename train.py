from transformers import BertModel, BertTokenizer, BertConfig
from model import BertForCopyRightNameMatching
from dataloader import TitleDataset
from torch.utils import data
import torch.optim as optim
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

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
        test_loss = 0.0
        model.zero_grad()
        t = tqdm(enumerate(training_generator, 0), total=len(training_generator))
        t.set_description("Epoch: {}, train_loss: {:.4f}, test_loss: {:.4f}".format(epoch+1, running_loss / 10, test_loss / len(test_generator)))
        # t.set_description("Loss: {:.4f}".format(running_loss / 10))
        for i, (input_ids, label) in t:

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

            if (i + 1) % 1000 == 0:     # Do validation
                with torch.set_grad_enabled(False):
                    all_pred_labels = []
                    all_labels = []
                    for input_ids, label in test_generator:
                        input_ids, label = input_ids.to(device), label.to(device)
                        outputs = model(input_ids, labels=label)
                        pred_labels = torch.max(outputs[1][0], 1)[1]

                        all_pred_labels += pred_labels
                        all_labels += label.cpu().numpy().tolist()

                        loss = outputs[0].squeeze()
                        test_loss += loss.item()
                t.set_description("Epoch: {}, train_loss: {:.4f}, test_loss: {:.4f}".format(epoch+1, running_loss / 10, test_loss / len(test_generator)))
                test_loss == 0.0
                acc = accuracy_score(all_labels, all_pred_labels)
                print("Acc: {}".format(acc))

            if (i + 1) % 100 == 0:    # print every 10 mini-batches
                # print('Epoch: {}, step: {}, loss: {}'.format(epoch + 1, i + 1, running_loss / 100))
                t.set_description("Epoch: {}, train_loss: {:.4f}, test_loss: {:.4f}".format(epoch+1, running_loss / 10, test_loss / len(test_generator)))
                running_loss = 0.0


    print("Done training")

if __name__ == "__main__":
    training_generator, test_generator, model = main()
    # input_ids, labels_1, labels_2 = next(iter(training_generator))
    train(training_generator=training_generator,
            test_generator=test_generator,
            model=model,
            device=device,
            max_epochs=10)
