"""
 Course:      Statistical Language Processing I - WS 2023/2024
 Assignment:  HW5, bigram model
 Author: Chi Kuan Lai

 Honor Code:  I pledge that this program represents my own work.
              No portion of this code has been copied from another source,
              and no portion of this code has been shared with others.
"""
import numpy as np
from unigram_model import UNK, Unigram


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

        """
        Process ONE training sent, which has already been
        partially preprocessed, where preprocessing includes:
            - replace OOV words with the UNK token
            - add BOS and EOS markers
.
        Return the result.

        Notes:
        - use self.vocab to detect OOV words

        :param sent: a sentence previously preprocessed for unigram, as list[str]
        :return: list[str] input sentence with OOV replaced, and markers inserted
        """
    def preprocess_train_sentence(self, sent):

        sent = [w if w in self.vocab else UNK for w in sent]
        clean_sent = [BOS] + sent + [EOS]
        
        return clean_sent
    


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
    def preprocess_test_sentence(self, sent):

        sent_lower = sent.lower()
        tokenize_sent = sent_lower.split() 
        replace_sent = [word if word in self.vocab else UNK for word in tokenize_sent]
        clean_sent = [BOS] + replace_sent + [EOS]
        
        return clean_sent


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
    def load_and_preprocess(self, train=True, corpus_file=None):

        if train:
            data = [self.preprocess_train_sentence(sent) for sent in self.unigram_model.training_data]
            self.training_data = data

        else:
            data = []

            with open(corpus_file, 'r', encoding='utf-8') as f:
                for sent in f:
                    processed_sent = self.preprocess_test_sentence(sent)
                    data.append(processed_sent)
            self.test_data = data


        """
        The vocabulary was established while training the unigram model.
        For convenience, also store the vocabulary here in class variable self.vocab.
        """
    def set_vocab(self):

        self.vocab = self.unigram_model.vocab


        """
        Assign each word in the vocabulary an index for access into the
        bigram probability matrix. Create 2 maps, one for the rows and one for the
        columns. The maps are dictionaries with vocabulary words as keys, and
        indices as values. The indices should start at 0 and each word should have
        a unique index. Append a BOS index in the row mapping, and an EOS index
        in the column mapping.
        Store the maps in self.row_idx_map and self.col_idx_map
        """
    def set_col_row_idx_maps(self):

        map1 = self.vocab+[BOS]
        map2 = self.vocab+[EOS]
        self.row_idx_map = {word: idx for idx, word in enumerate(map1)}
        self.col_idx_map= {word: idx for idx, word in enumerate(map2)}


        """
        Create a 2D matrix (np.array) containing the bigram counts in self.training_data.
        Use self.row_idx_map and self.col_idx_map to access cells in the matrix.
        For example:
            If the row index for the token 'cats' is 3, and the column index for the
            token 'like' is 1, the count for the bigram 'cats like' is stored in cell [3][1]

        Store the matrix in self.counts
        """
    def count_bigrams(self):

        row = len(self.row_idx_map) 
        col = len(self.col_idx_map)
        self.counts = np.zeros((row, col)) #create a numpy array with all 0
        for sentence in self.training_data:
            for i in range(len(sentence) - 1):
                word = sentence[i] 
                next_word = sentence[i + 1]

                if word in self.row_idx_map and next_word in self.col_idx_map:
                    row_idx = self.row_idx_map[word] #get the index of current word
                    col_idx = self.col_idx_map[next_word] #get the index of next word
                    self.counts[row_idx, col_idx] += 1


        """
        Apply Add-k smoothing to self.counts.

        :param k: k-value to add, defaults to 1
        """
    def smooth(self, k=1.0):

        self.counts = [x+k for x in self.counts]

        """
        Use the bigram counts in self.counts to create a new matrix
        containing the bigram probabilities.
        Store the result in self.probs
        """
    def calc_probs(self):

        total_count = np.sum(self.counts,axis=0) 
        
        array_total_count = np.array([[x] for x in total_count]) # make it from [1,2,3] to [[1],[2],[3]]
        self.probs = self.counts/array_total_count
 

        """
        Create a new matrix containing the log 10 probabilities of self.probs.
        Store the result in self.logprobs
        """
    def calc_log10_probs(self):

        self.logprobs = np.log10(self.probs)


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
    def train(self, train_file, k=1.0):

        self.unigram_model.train(train_file)
        self.set_vocab()
        self.load_and_preprocess(train=True)
        self.set_col_row_idx_maps()
        self.count_bigrams()
        self.smooth(k)
        self.calc_probs()
        self.calc_log10_probs()


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
    def generate_sentence(self):

        gen_sent = [BOS]
        current_word = BOS

        while current_word != EOS:
           idx_cword = self.row_idx_map[current_word]
           next_word = rng.choice(list(self.col_idx_map.keys()), p=self.probs[idx_cword])
           gen_sent.append(next_word)
           current_word = next_word

        return gen_sent

        """
        Returns the log 10 probability of ONE sentence.

        :param sent: list[str], a fully preprocessed sentence
        :return: the log10 probability of sent
        """
    def calc_sent_logprob(self, sent):

        log_prob = 0.0

        for i in range(len(sent) - 1):
           current_word = sent[i] 
           next_word = sent[i + 1]
           if current_word in self.row_idx_map and next_word in self.col_idx_map:
               row_idx = self.row_idx_map[current_word] #get the index from row
               col_idx = self.col_idx_map[next_word] #get the index from col
               log_prob += self.logprobs[row_idx, col_idx] #find the probability by the row index and column index

        return log_prob


        """
        Calculate the perplexity of the preprocessed test sentences
        according to the trained bigram model.

        :param test_sentences: list of fully preprocessed test sentences
        :return: the perplexity of the test sentences according to the trained bigram model
        """
    def calc_perplexity(self, test_sentences):

        total_log_prob = sum(self.calc_sent_logprob(sent) for sent in test_sentences)
        total_bigrams = sum(len(sent) - 1 for sent in test_sentences)


        perplexity = 10 ** (-total_log_prob / total_bigrams)
       
        return perplexity

        """
        Test both the unigram model (self.unigram_model) and bigram model
        on the data in test_file.

        Return unigram perplexity and bigram perplexity of the test data.

        Hint: self.unigram_model has already been trained, and has its own test method.

        :param test_file: test sentence strings, one per line
        :return: unigram perplexity, bigram perplexity of the test data
        """
    def test(self, test_file):

        self.load_and_preprocess(train=False, corpus_file=test_file)
        bi_perplexity = self.calc_perplexity(self.test_data)

        uni_perplexity = self.unigram_model.test(test_file)
        return uni_perplexity,bi_perplexity


    """
    Train a bigram model using the Sherlock Holmes training data, using Add-k smoothing.
    Generate and print 2 sentences.
    Calculate and print the perplexity of the Sherlock Holmes test data,
        using a unigram model and the bigram model.

    Experiment with different values for k, and use the best value for training.
    """
def main():
    bigram = Bigram()
    train_data="Data/SherlockHolmes-train.txt"
    test_data="Data/SherlockHolmes-test.txt"
    
    #Find the best k value by using For Loop
    """
    best_k = 0.000
    best_perplexity = float('inf')

    for i in np.arange(0.001, 1.000, 0.001):
        bigram.train(train_data, k=i)
        perplexity = bigram.test(test_data)[1]
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            best_k = i

    print("Best k:", best_k)
    #Best k=0.008
    """
    bigram.train(train_data, k=0.008)

    i=0
    while i < 2:
       generated_sentence = bigram.generate_sentence()
       print(generated_sentence)
       i += 1

    perp = bigram.test(test_data)
    print("Perplexity of Unigram and Bigram: ",perp)


if __name__ == '__main__':
    main()
