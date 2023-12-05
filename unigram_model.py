'''
 * Course:      Parsing Suse23
 * Assignment:  Homework 05
 * Author:      Yifei Chen
 * Description: Unigram model training
 *
 * Honour Code:  I pledge that this program represents my own work.
 *  I received help from no one in designing and debugging my program.
'''


import numpy as np

# define constants
UNK = "<UNK>"

# random number generator
rng = np.random.default_rng()


class Unigram:

    def __init__(self):

        # Class variables will be given values in the class methods
        self.training_data = None  # list[list[str]]
        self.test_data = None  # list[list[str]]
        self.vocab = None  # lst[str]
        self.counts = None  # {word : freq}
        self.col_idx_map = None  # {word : index}
        self.probs = None  # np.array of length len(vocab)
        self.logprobs = None  # np.array of length len(vocab)

    def preprocess_sentence(self, sent, train=True):
        """
        Process ONE sent, where preprocessing includes:
            - tokenization (splitting on whitespace)
            - lowercasing tokens
            - If train==False (sent is a test sentence),
                also replace OOV words with the UNK token
        Return the preprocessed sentence as list[str]

        Notes:
        - use self.vocab to detect OOV words in the test data

        :param sent: sentence to process, as a string
        :param train: if True, sent is a training sentence, otherwise sent is a test sentence.
        :return: the preprocessed sentence, as list[str]
        """
        # tokenisation, split on whitespace
        sent_lower = sent.lower()
        tokens = sent_lower.split()

        if not train:
            tokens = [token if token in self.vocab else UNK for token in tokens]

        return tokens

    def load_and_preprocess(self, corpus_file, train=True):
        """
        Read corpus_file, which contains one sentence per line.
        Each sentence is preprocessed by preprocess_sentence().
        If train==True, corpus_file contains training data. In this case, the preprocessed
        sentences (represented as a list[list[str]]), are stored in the class variable
        self.training_data.
        If train==False, corpus_file contains test data. In this case, the preprocessed
        sentences (represented as a list[list[str]]), are stored in the class variable
        self.test_data.

        :param corpus_file: file containing one sentence per line
        :param train: boolean, if True, corpus_file contains training data, otherwise test data
        """
        data = []  # List to store preprocessed sentences
        with open(corpus_file, 'r', encoding='utf-8') as file:
            for line in file: # iterates over each line
                # Preprocess each sentence using preprocess_sentence
                preprocessed_sentence = self.preprocess_sentence(line.strip(), train) # delete spaces, pass lines and var[train]
                data.append(preprocessed_sentence) # Appends the preprocessed sentence to the data list

        # Store the preprocessed sentences based on whether it's training/testing data
        if train:
            self.training_data = data
        else:
            self.test_data = data

    def count_unigrams(self):
        """
        Create a dictionary of unigram frequencies in the training data
         contained in self.training_data.
        The keys in the dictionary are words, and the values are
        the number of times the word appears in the training data.
        Store the frequency dictionary in self.counts.
        """
        self.counts = {}  # Initialize an empty dictionary to store freq

        # Iterate through preprocessed sentences in the training data
        for sentence in self.training_data:
            # Count unigram frequencies in each sentence
            for word in sentence:
                if word in self.counts:
                    self.counts[word] += 1 # count 1 if in training data
                else:
                    self.counts[word] = 1 # else remain 1

    def remove_low_freq_words(self): # ???
        """
        Replace low-frequency words in self.counts with the UNK token.
        Remove any word with count == 1 from self.counts,
        and add its count to the count of the UNK token.
        Store the result back into self.counts.
        """
        unk_count = 0  # Initialize count for the UNK token

        # Identify and replace low-frequency words with UNK
        for word, count in list(self.counts.items()):  # Use list() to create a copy for safe iteration
            if count == 1:
                unk_count += count  # Add count of low-frequency word to UNK count
                del self.counts[word]  # remove freq count = 1 from self.counts
                self.counts[UNK] = unk_count  # Update UNK count


    def set_vocab(self):
        """
        Set the class variable self.vocab to the SORTED list of words
        in the vocabulary. Sorting is not required for the code to work,
        but otherwise some tests may fail.
        """
        self.vocab = sorted(self.counts.keys())

    def set_col_idx_map(self):
        """
        Assign each word in the vocabulary a unique index for access into the
        unigram probability array.
        The mapping is a dictionary with vocabulary words as keys, and
        indices as values. The indices should start at 0 and each word should have
        a unique index.
        Store the mapping in self.col_idx_map
        """
        self.col_idx_map = {}  # Initialize the dictionary for mapping

        # Assign unique indices to vocabulary words
        for index, word in enumerate(self.vocab):
            self.col_idx_map[word] = index

    def calc_probs(self):
        """
        Use self.vocab, self.counts and self.col_idx_map to fill an np.array of
        length len(vocab) with unigram probabilities.
        For example, if the word "the" was assigned the index 3, the generated
        array at index 3 should contain the probability of the word "the".
        Store the array in the class variable self.probs
        """
        total_word_count = sum(self.counts.values())  # Total count of all words in the training data

        # Initialize the array for unigram probabilities
        self.probs = np.zeros(len(self.vocab))

        # Calculate and fill unigram probabilities in the array
        for word, count in self.counts.items():
            index = self.col_idx_map[word]
            self.probs[index] = count / total_word_count

    def calc_log10_probs(self):
        """
        Create a new np.array, stored in the class variable self.logprobs,
        that contains the probabilities in self.probs converted to
        log base 10 probabilities.
        """
        # Check if self.probs is not None
        if self.probs is not None:
            # Calculate log base 10 probabilities
            self.logprobs = np.log10(self.probs)
        else:
            raise ValueError("Unigram probabilities (self.probs) have not been calculated.")

    def train(self, train_file):
        """
        Using the methods above, train the unigram model on the
        data in train_file.
        Training includes loading and preprocessing the data,
        removing low-frequency words, and
        calculating all counts, mappings, and probabilities.

        :param train_file: file with training data, one sentence per line
        """
        # Load and preprocess training data
        self.load_and_preprocess(train_file, train=True)

        # Count unigrams in the training data
        self.count_unigrams()

        # Remove low-frequency words and update counts
        self.remove_low_freq_words()

        # Set vocabulary and mappings
        self.set_vocab()
        self.set_col_idx_map()

        # Calculate unigram probabilities and log probabilities
        self.calc_probs()
        self.calc_log10_probs()

    def generate_sentence(self, sent_len):
        """
        Use the probabilities in self.probs to generate
        a sentence of length sent_len, where words are chosen according
        to their unigram probabilities. That is, words with higher probabilities
        are more likely to be chosen than those with low probabilities.

        Notes:

        - self.probs forms a probability distribution
        - this task is easily accomplished with the choice() function of
            random number generator object (rng) defined at the top of this file.

        :param sent_len: length of the sentence to generate
        :return: a list of words ['w1', ..., 'wn'] generated from the model
        """
        if self.probs is None:
            raise ValueError("Unigram probabilities (self.probs) have not been calculated.")

        # Use the choice() function to generate a sentence based on probabilities
        generated_sentence = rng.choice(self.vocab, sent_len, p=self.probs)

        return list(generated_sentence)

    def calc_sent_logprob(self, sent):
        """
        Return the log probability of ONE sentence.

        :param sent: list[str] a sentence represented as a list of tokens
        :return: the log10 probability of sent
        """
        if self.logprobs is None:
            raise ValueError("Log probabilities (self.logprobs) have not been calculated.")

        # Calculate the log probability of the sentence by summing log probabilities of individual words
        sent_logprob = sum(self.logprobs[self.col_idx_map.get(word, self.col_idx_map[UNK])] for word in sent)

        return sent_logprob

    def calc_perplexity(self, test_sentences):
        """
        Return the perplexity of the given test sentences.
        Use log10 probabilities to prevent underflow.

        :param test_sentences: list[list[str]] of preprocessed test sentences
        :return: the perplexity of the given test sentences
        
        if self.logprobs is None:
            raise ValueError("Log probabilities (self.logprobs) have not been calculated.")
        """
        
        total_logprob = 0.0
        total_words = 0

        # Calculate the total log probability and total number of words in the test set
        for sent in test_sentences:
            total_logprob += self.calc_sent_logprob(sent)
            total_words += len(sent)

        # Calculate perplexity using the total log probability and total number of words
        perplexity = 10 ** (-total_logprob / total_words)

        return perplexity

    def test(self, test_file):
        """
        Return the perplexity of the test sentences in test_file.

        :param test_file: file containing test data, one sentence string per line
        :return: the perplexity of the test sentences
        """
        # Load and preprocess test data
        self.load_and_preprocess(test_file, train=False) # (Error 1: not test_data)

        # Calculate and return the perplexity of the test sentences
        perplexity = self.calc_perplexity(self.test_data)
        return perplexity


def main():
    """
    Train a unigram model using the Sherlock Holmes training data.
    Generate and print 5 sentences of length 20.
    Calculate the perplexity of the Sherlock Holmes test data.
    """

    # Create an instance of the Unigram class
    model = Unigram()

    # Train the unigram model on the Sherlock Holmes training data
    model.train("Data/SherlockHolmes-train.txt")

    # Generate and print 5 sentences of length 20
    for _ in range(5):
        generated_sentence = model.generate_sentence(20)
        print(generated_sentence)

    # Calculate the perplexity of the Sherlock Holmes test data
    perplexity = model.test("Data/SherlockHolmes-test.txt")
    print("Perplexity of Sherlock Holmes test data:", perplexity)


if __name__ == '__main__':
    main()