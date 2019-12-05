from torch.utils import data
from keras.preprocessing.sequence import pad_sequences
import torch
import pandas as pd

class TitleDataset(data.Dataset):
    def __init__(self, data_path, tokenizer, max_full_text_len):
        super(TitleDataset).__init__()
        self.data = pd.read_csv(data_path, sep='\t')
        self.ids = [i for i in range(len(self.data))]
        self.max_full_text_len = max_full_text_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        sample = self.data.loc[[id]]
        cr = sample['cr'].values[0]
        name = sample['name'].values[0]
        label = sample['labels'].values[0]

        cr_encoded = self.tokenizer.encode("[CLS] " + str(cr))
        name_encoded = self.tokenizer.encode(" [SEP] " + str(name) + " [SEP]")

        full_text = pad_sequences([cr_encoded + name_encoded],
                                                maxlen= self.max_full_text_len,
                                                dtype="long",
                                                truncating="post",
                                                padding="post")

        input_ids = torch.tensor(full_text).unsqueeze(0)
        if label is False:
            label_tensor = torch.tensor([1]).unsqueeze(0)
        else:
            label_tensor = torch.tensor([0]).unsqueeze(0)

        return input_ids, label_tensor

if __name__ == "__main__":
    pass
