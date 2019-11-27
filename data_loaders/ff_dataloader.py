from utils import word2ngram
from tqdm import tqdm

class Generator(object):
    """
    Dataframe includes these following fields, respectively:
    - id, company_key, company_name, google_name, score, class, company_name_non_space, google_name_non_space
    """
    def __init__(self, data, num_character=[2, 3], num_character_position=5, num_words=[1, 2], num_word_position=3):
        self.data = data
        self.num_character = num_character
        self.num_character_position = num_character_position
        self.num_words = num_words
        self.num_word_position = num_word_position


    def score_ngram_character(self):

        for n in self.num_character:
            # Create n-gram-character columns
            for i in range (0, self.num_character_position):
                self.data['n-gram-character-{}-{}-gram'.format(i, n)] = 0.0

            # Lopp through each row and score for each n-gram-character
            # Score 1 if match, 0, otherwise
            for index, row in self.data.iterrows():
                ngram_company_name, num_company_name = word2ngram(row['company_name_non_space'])
                ngram_google_name, num_google_name = word2ngram(row['google_name_non_space'])

                max_count = self.num_character_position if self.num_character_position <= min(
                    num_company_name, num_google_name) else min(num_company_name, num_google_name)

                for i in range(0, self.num_character_position):
                    if i < max_count:
                        self.data.at[index, 'n-gram-character-{}-{}-gram'.format(
                            i, n)] = 1 if ngram_company_name[i] == ngram_google_name[i] else 0
                    else:
                        self.data.at[index,
                                    'n-gram-character-{}-{}-gram'.format(i, n)] = 0

                self.data.at[index, 'matched-all-ngram-character'] = 1.0 if row['company_name_non_space'] == row['google_name_non_space'] else 0.0

        print("Done generating feature for n_gram character")

    def score_ngram_word(self):
        for n in self.num_words:

            for i in range(0, self.num_word_position):
                self.data['n-gram-word-{}-{}-gram'.format(i, n)] = 0.0

                # Lopp through each row and score for each n-gram-character
                # Score 1 if match, 0, otherwise
                for index, row in self.data.iterrows():

                    ngram_company_name, num_company_name = word2ngram(row['company_name'], token='word')
                    ngram_google_name, num_google_name = word2ngram(row['google_name'], token='word')

                    max_count = self.num_word_position if self.num_word_position <= min(
                        num_company_name, num_google_name) else min(num_company_name, num_google_name)



                    for i in range(0, self.num_word_position):
                        if i < max_count:
                            self.data.at[index, 'n-gram-word-{}-{}-gram'.format(
                                i, n)] = 1 if ngram_company_name[i] == ngram_google_name[i] else 0
                        else:
                            self.data.at[index,
                                        'n-gram-word-{}-{}-gram'.format(i, n)] = 0

                    self.data.at[index, 'matched-all-ngram-word'] = 1.0 if row['company_name'] == row['google_name'] else 0.0

            print("Done generating feature for n_gram word")

    def play(self):
        print('Begin to generate n_gram character score')
        self.score_ngram_character()
        print('Begin to generate n_gram word score')
        self.score_ngram_word()

    def return_data(self):
        return self.data

def main():
    print(word2ngram('kantangparawoodcompanylimited'))

if __name__ == "__main__":
    main()
