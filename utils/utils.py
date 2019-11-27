import pandas as pd
from tqdm import tqdm
import glob

def read_data(data_path):
    return 0

def main():
    true_match = 0
    files = glob.glob('data/*.xlsx')

    for file in files:
        print(file)
        data = pd.read_excel(file)
        for _, row in tqdm(data.iterrows(), total=data.shape[0]):
            if row['true match?'] == 1:
                true_match += 1
            else:
                pass
    return true_match
if __name__ == "__main__":
    true_match = main()
    print(true_match)
