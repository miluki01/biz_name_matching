import torch
import numpy as np
from utils import word2ngram, remove_pipeline
from torch.autograd import Variable
import torch.nn.functional as F

num_character=[2]
num_character_position=5
num_words=[1]
num_word_position=2


def string2vec(s1, s2, stop_words, num_character=[2], num_character_position=5, num_words=[1], num_word_position=2):
    vector_character = np.zeros(len(num_character) * num_character_position)
    vector_word = np.zeros(len(num_words) * num_word_position)

    post_s1_non_space = remove_pipeline(s1, stop_words=stop_words, keep_space=False)
    post_s2_non_space = remove_pipeline(s2, stop_words=stop_words, keep_space=False)
    ngram_s1_character, num_s1_character = word2ngram(post_s1_non_space)
    ngram_s2_character, num_s2_character = word2ngram(post_s2_non_space)

    post_s1_with_space = remove_pipeline(s1, stop_words=stop_words, keep_space=True)
    post_s2_with_space = remove_pipeline(s2, stop_words=stop_words, keep_space=True)
    ngram_s1_word, num_s1_word = word2ngram(post_s1_with_space, token='word')
    ngram_s2_word, num_s2_word = word2ngram(post_s2_with_space, token='word')

    max_count_character = num_character_position if num_character_position <= min(
                num_s1_character, num_s2_character) else min(num_s1_character, num_s2_character)

    max_count_word = num_word_position if num_word_position <= min(
                num_s1_word, num_s2_word) else min(num_s1_word, num_s2_word)

    for i in range(vector_character.shape[-1]):
        if i < max_count_character:
            vector_character[i] = 1 if ngram_s1_character[i] == ngram_s2_character[i] else 0
        else:
            vector_character[i] = 0

    for i in range(vector_word.shape[-1]):
        if i < max_count_word:
            vector_word[i] = 1 if ngram_s1_word[i] == ngram_s2_word[i] else 0
        else:
            vector_word[i] = 0

    match_all_character = np.ones(1) if post_s1_non_space == post_s2_non_space else np.zeros(1)
    match_all_word = np.ones(1) if post_s1_with_space == post_s2_with_space else np.zeros(1)

    # Concat 2 vector
    vector = np.concatenate((vector_character, vector_word, match_all_character, match_all_word))

    return np.resize(vector, (1, len(num_character) * num_character_position + len(num_words) * num_word_position + 2))


def main():
    import sys
    from models import NameMatchingModel

    try:
        with open('data/Common_words_UPDATED.txt', 'r') as f:
            stop_words = [word.strip() for word in f.readlines()]

    except IOError:
        sys.exit('stopword file is not found')

    model = NameMatchingModel()

    model.load_state_dict(torch.load('saved_model/name_matching_100_scale.pth'))
    vector = string2vec('Rumah Gelato Frozen Treats', 'RUMAH MAKAN AYAH BUNDA', stop_words=stop_words)

    tensor_vector = Variable(torch.FloatTensor(vector), requires_grad=False)
    print(tensor_vector)

    with torch.no_grad():
        result = model(tensor_vector)

    print((torch.exp(result) * 100).type(torch.IntTensor).numpy()[0].tolist()[1])


if __name__ == '__main__':
    main()
