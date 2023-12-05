'''
 * Course:      Parsing SuSe23
 * Assignment:  Homework 05
 * Author:      Yifei Chen
 * Description: Bigram model training
 *
 * Honour Code:  I pledge that this program represents my own work.
 *  I received help from no one in designing and debugging my program.
'''


import numpy as np
from unigram_model import UNK, Unigram

# define constants
BOS = "<s>"
EOS = "</s>"

# random number generator
rng = np.random.default_rng()


class Bigram():
    def __init__(self):

        # train a unigram model to get vocab, etc
        self.unigram_model = Unigram()

        # Class variables will be given values in the class methods
        self.training_data = None  # list[list[str]]
        self.test_data = None  # list[list[str]]
        self.vocab = None  # lst[str]
        self.row_idx_map = None  # {word : index}
        self.col_idx_map = None  # {word : index}
        self.counts = None  # 2D count matrix (np.array), rows are w1, cols are w2
        self.probs = None  # 2D prob matrix (np.array), rows are w1, cols are w2
        self.logprobs = None  # 2D log10 prob matrix (np.array), rows are w1, cols are w2

    def preprocess_train_sentence(self, sent):
        """
        Process ONE training sent, which has already been partially preprocessed, where preprocessing includes:
            - replace OOV words with the UNK token
            - add BOS and EOS markers
        Return the result.

        Notes:
        - use self.vocab to detect OOV words

        :param sent: a sentence previously preprocessed for unigram, as list[str]
        :return: list[str] input sentence with OOV replaced, and markers inserted
        """
        # Replace OOV words with UNK
        processed_sent = [word if word in self.vocab else UNK for word in sent]

        # Add BOS and EOS markers
        processed_sent2 = [BOS] + processed_sent + [EOS]

        return processed_sent2

    def preprocess_test_sentence(self, sent):
        """
        Process ONE test sent, where preprocessing includes:
            - tokenization (splitting on whitespace)
            - lowercasing tokens
            - replace OOV words with the UNK token
            - add BOS and EOS markers

        Notes:
        - use self.vocab to detect OOV words in the test data

        :param sent: sentence to process, as a string
        :return: the preprocessed sentence, as list[str]
        """
        
        # Tokenization, lowercasing
        preprocessed_sent = sent.lower().split()
        
        # Replace OOV words with UNK
        preprocessed_sent3 = [word if word in self.vocab else UNK for word in preprocessed_sent]

        # Add BOS and EOS markers
        processed_sent = [BOS] + preprocessed_sent3 + [EOS]

        return processed_sent        

    def load_and_preprocess(self, train=True, corpus_file=None):
        """
        Load and preprocess training or testing data.

        If train==True, the training data was already partially preprocessed while
        training the unigram model. The unigram training data can be found in
        self.unigram_model.training_data. In this case, use preprocess_train_sentence()
        to finish preprocessing.
        Store the result in the class variable self.training_data.

        If train==False, corpus_file contains test data, with one sentence per line.
        In this case, use preprocess_test_sentence() for preprocessing each sentence.
        Store the result [list[list[str]] in the class variable self.test_data

        :param train: if True, further process unigram training data; otherwise process test data in corpus_file
        :param corpus_file: file containing one sentence string per line if train=False
        """
        if train:
            # Further process unigram training data
            self.training_data = [self.preprocess_train_sentence(sent) for sent in self.unigram_model.training_data]
        else:
            # Process test data from corpus_file
            with open(corpus_file, 'r') as file:
                self.test_data = [self.preprocess_test_sentence(line.strip()) for line in file]        

    def set_vocab(self):
        """
        The vocabulary was established while training the unigram model.
        For convenience, also store the vocabulary here in class variable self.vocab.
        """
        self.vocab = self.unigram_model.vocab

    def set_col_row_idx_maps(self):
        """
        Assign each word in the vocabulary an index for access into the
        bigram probability matrix. Create 2 maps, one for the rows and one for the
        columns. The maps are dictionaries with vocabulary words as keys, and
        indices as values. The indices should start at 0 and each word should have
        a unique index. Append a BOS index in the row mapping, and an EOS index
        in the column mapping.
        Store the maps in self.row_idx_map and self.col_idx_map
        """

        # Create row and column index maps
        d1 = self.vocab + [BOS]
        d2 = self.vocab + [EOS]
        self.row_idx_map = {word: idx for idx, word in enumerate(d1)}
        self.col_idx_map = {word: idx for idx, word in enumerate(d2)}     


    def count_bigrams(self):
        """
        Create a 2D matrix (np.array) containing the bigram counts in self.training_data.
        Use self.row_idx_map and self.col_idx_map to access cells in the matrix.
        For example:
            If the row index for the token 'cats' is 3, and the column index for the
            token 'like' is 1, the count for the bigram 'cats like' is stored in cell [3][1]

        Store the matrix in self.counts
        """
        num_rows = len(self.row_idx_map)
        num_cols = len(self.col_idx_map)

        # Initialize counts matrix with zeros
        self.counts = np.zeros((num_rows, num_cols), dtype=int)

        # Count bigrams in training data
        for sentence in self.training_data:
            for i in range(len(sentence) - 1):
                row_idx = self.row_idx_map[sentence[i]]
                col_idx = self.col_idx_map[sentence[i + 1]]
                self.counts[row_idx][col_idx] += 1

    def smooth(self, k=1.0):
        """
        Apply Add-k smoothing to self.counts.

        :param k: k-value to add, defaults to 1
        """
        # Add k to all counts
        self.counts = self.counts + k

        '''
        # Calculate row-wise probabilities after smoothing
        row_sums = np.sum(self.counts, axis=1)
        self.probs = self.counts / row_sums[:, np.newaxis]
        NOT KEEP IT, WRONG CODES'''


    def calc_probs(self):
        """
        Use the bigram counts in self.counts to create a new matrix
        containing the bigram probabilities.
        Store the result in self.probs
        """
        row_sums = np.sum(self.counts, axis = 0)
        self.probs =  self.counts / row_sums.reshape(-1, 1)

        '''
        # Calculate row-wise probabilities
        row_sums = np.sum(self.counts, axis=0)
        self.probs = self.counts / row_sums[:, np.newaxis]
        '''

    def calc_log10_probs(self):
        """
        Create a new matrix containing the log 10 probabilities of self.probs.
        Store the result in self.logprobs
        """
        # Avoid log(0) by adding a small value
        epsilon = 1e-10

        # Calculate log10 probabilities
        self.logprobs = np.log10(self.probs + epsilon)

    def train(self, train_file, k=1.0):
        """
        Using the methods above, train the bigram model on the
        data in corpus_file.
        Training includes:
            - training self.unigram_model using train_file
            - setting the vocabulary (same as unigram vocabulary)
            - loading and further preprocessing the training data,
            - calculating all counts (including smoothing), mappings, and probabilities.

        :param train_file: file with training data, one sentence string per line
        :param k: smoothing factor
        """
        # Train unigram model
        self.unigram_model.train(train_file)

        # Set vocabulary based on unigram model
        self.set_vocab()

        # Load and preprocess training data
        self.load_and_preprocess(train=True)

        # Matrix maps
        self.set_col_row_idx_maps()

        # Calculate bigram counts with smoothing
        self.count_bigrams()

        # Apply Add-k smoothing
        self.smooth(k)

        # Calculate bigram probabilities
        self.calc_probs()

        # Calculate log10 probabilities
        self.calc_log10_probs()

    def generate_sentence(self):
        """
        Generate a sentence with probabilities according to the distributions
        in self.probs (don't use log probabilities here).
        That is, the next word is chosen using the probability distribution
        of the current word.
        The returned sentence is a list of words that
        starts with BOS and ends when EOS is generated.
        The choice() function of random number generator object (rng) is helpful.

        :return: a sentence as list[str], generated using self.probs
        """
        sentence = [BOS]
        current_word = BOS

        while current_word != EOS:
            # Get row index for the current word
            row_idx = self.row_idx_map[current_word]

            # Sample the next word based on the probability distribution
            next_word = rng.choice(list(self.col_idx_map.keys()), p=self.probs[row_idx])

            # Update the sentence and current word
            sentence.append(next_word)
            current_word = next_word

        return sentence

    def calc_sent_logprob(self, sent):
        """
        Returns the log 10 probability of ONE sentence.

        :param sent: list[str], a fully preprocessed sentence
        :return: the log10 probability of sent
        """
        log_prob = 0.0

        for i in range(len(sent)-1):
            # Get row and column indices for the bigram
            row_idx = self.row_idx_map[sent[i]]
            col_idx = self.col_idx_map[sent[i + 1]]
            # Add log probability of the bigram to the total log probability
            log_prob += self.logprobs[row_idx, col_idx]
        
        return log_prob

    def calc_perplexity(self, test_sentences):
        """
        Calculate the perplexity of the preprocessed test sentences
        according to the trained bigram model.

        :param test_sentences: list of fully preprocessed test sentences
        :return: the perplexity of the test sentences according to the trained bigram model
        """
        total_logprob = 0.0
        total_words = 0

        for sent in test_sentences:
            total_logprob += self.calc_sent_logprob(sent)
            total_words += len(sent)-1
        
        perplexity = 10 ** (-total_logprob / total_words)

        return perplexity

        '''
        total_logprob = 0.0
        total_words = 0

        for sent in test_sentences:
            total_logprob += self.calc_sent_logprob(sent)
            total_words += len(sent)

        # Calculate average log probability per word
        avg_logprob = total_logprob / total_words

        # Calculate perplexity using the formula 10^(-1/N * logprob)
        perplexity = 10 ** (-1 / total_words * avg_logprob)

        return perplexity
        '''

    def test(self, test_file):
        """
        Test both the unigram model (self.unigram_model) and bigram model
        on the data in test_file.

        Return unigram perplexity and bigram perplexity of the test data.

        Hint: self.unigram_model has already been trained, and has its own test method.

        :param test_file: test sentence strings, one per line
        :return: unigram perplexity, bigram perplexity of the test data
        """
        # Test the unigram model
        unigram_perplexity = self.unigram_model.test(test_file)

        # Load and preprocess test data for the bigram model
        self.load_and_preprocess(train=False, corpus_file=test_file)

        # Calculate bigram perplexity
        bigram_perplexity = self.calc_perplexity(self.test_data)

        return unigram_perplexity, bigram_perplexity


def main():
    """
    Train a bigram model using the Sherlock Holmes training data, using Add-k smoothing.
    Generate and print 2 sentences.
    Calculate and print the perplexity of the Sherlock Holmes test data,
        using a unigram model and the bigram model.

    Experiment with different values for k, and use the best value for training.
    """


    # Train the unigram model
    unigram_model = Unigram()
    train_file_unigram = "Data/SherlockHolmes-train.txt"
    unigram_model.train(train_file_unigram)

    # Train the bigram model
    bigram_model = Bigram()
    train_file_bigram = "Data/SherlockHolmes-train.txt"
    k_values = [0.1, 0.5, 1.0, 1.5]  # Experiment with different values for k

    best_k = None
    best_perplexity = float('inf')

    for k in k_values:
        bigram_model.train(train_file_bigram, k=k)

        # Generate and print 2 sentences
        sentence1 = bigram_model.generate_sentence()
        sentence2 = bigram_model.generate_sentence()

        # Calculate and print perplexity of Sherlock Holmes test data
        test_file = "Data/SherlockHolmes-test.txt"
        unigram_perplexity, bigram_perplexity = bigram_model.test(test_file)

        # Update best k if current perplexity is lower
        if bigram_perplexity < best_perplexity:
            best_perplexity = bigram_perplexity
            best_k = k

    print(f"Sentence 1: {' '.join(sentence1)}")
    print(f"Sentence 2: {' '.join(sentence2)}\n")
    print(f"Unigram Perplexity: {unigram_perplexity:.4f}")
    print(f"Bigram Perplexity (k={k}): {bigram_perplexity:.4f}")
    print(f"Best k: {best_k}")


if __name__ == '__main__':
    main()
