import pandas as pd
from tqdm import tqdm
import glob
import yaml
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
import sys

def parse_config(file_path):
    return yaml.load(open(file_path), Loader=yaml.FullLoader)

def read_data(data_path):
    return 0

def remove_stopwords(string, stop_words):
    tokens, _ = word2ngram(string, n=1, token='word')
    result = [word for word in tokens if word.lower() not in stop_words]

    return result

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def nltk2wn_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatize_sentence(sentence, lemmatizer = WordNetLemmatizer()):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if tag is None:
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))

    return " ".join(res_words)

def remove_inside_parentheses(string):
    regex = re.compile("\(.*\)|\s-\s.*")
    result = re.sub(regex, "", string)

    return result

def remove_none_alphabet(string, stop_words, keep_space=False):
    regex = re.compile('\s*[^A-Za-z]+\s*')
    string = re.sub(regex, ' ', string)

    result = remove_stopwords(string, stop_words)

    if keep_space is False:
        return "".join(result)

    else:
        return " ".join(result)

def remove_pipeline(string, stop_words, keep_space=False):
    result = remove_inside_parentheses(lemmatize_sentence(string))
    result = remove_none_alphabet(lemmatize_sentence(result), stop_words=stop_words, keep_space=keep_space)

    return result.lower()

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def word2ngram(text, n=3, token='character'):
    if token == 'character':
        ngram = ["".join(j) for j in zip(*[text[i:] for i in range(n)])]
        return ngram, len(ngram)

    elif (token == 'word') and (text is not np.nan):
        ngram = nltk.word_tokenize(text)
        return ngram, len(ngram)

    else:
        return [], 0

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
