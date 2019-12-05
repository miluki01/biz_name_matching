import pandas as pd
from tqdm import tqdm
import glob
import multiprocessing as mp
from sklearn.model_selection import train_test_split
import os


def read_data(data_path):
    return 0

def get_sample(file):
    data = pd.read_excel(file)
    matched = []
    unmatched = []

    matched_data = data[data['true match?'] == 1]
    unmatched_data = data[data['true match?'] == 0]
    true_match = len(matched_data)
    sample_unmatched_data = unmatched_data.sample(n=true_match)

    for _, row in tqdm(matched_data.iterrows(), total=matched_data.shape[0]):
        cr = row['matched_cr']
        name = row['name']
        matched.append((cr, name))

    for _, row in tqdm(sample_unmatched_data.iterrows(), total=sample_unmatched_data.shape[0]):
        cr = row['matched_cr']
        name = row['name']
        unmatched.append((cr, name))


    return (matched, unmatched)

def main():
    files = glob.glob("data/*.xlsx")
    all_matched = []
    all_unmatched = []
    for file in files:
        matched, unmatched = get_sample(file)
        all_matched += matched
        all_unmatched += unmatched
    pd.DataFrame.from_records(all_matched, columns=['cr','name']).to_csv('matched.csv', index=False, sep='\t')
    pd.DataFrame.from_records(all_unmatched, columns=['cr','name']).to_csv('unmatched.csv', index=False, sep='\t')

    print('Done')

def merge_data():
    matched = pd.read_csv('matched.csv', sep='\t')
    matched['labels'] = True
    unmatched = pd.read_csv('unmatched.csv', sep='\t')
    unmatched['labels'] = False

    data = pd.concat([matched, unmatched])

    train, test = train_test_split(data, test_size=0.2, random_state=42)

    train.to_csv(os.path.join('data', 'train.csv'), index=False, sep='\t')
    test.to_csv(os.path.join('data', 'test.csv'), index=False, sep='\t')

    print("Done merge data")
if __name__ == "__main__":
    merge_data()
