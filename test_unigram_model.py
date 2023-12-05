import unittest

from unigram_model import *


class UnigramTestCase(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.unigram = Unigram()

    @classmethod
    def tearDown(self):
        self.unigram = None

    def test_preprocess_sentence1(self):
        sent = "This  is \t EASY !"
        expected = ["this", "is", "easy", "!"]
        actual = self.unigram.preprocess_sentence(sent, train=True)
        self.assertListEqual(expected, actual)

    def test_preprocess_sentence2(self):
        self.unigram.vocab = ["this", "is", "it"]
        sent = "This  is EASY !"
        expected = ["this", "is", UNK, UNK]
        actual = self.unigram.preprocess_sentence(sent, train=False)
        self.assertListEqual(expected, actual)

    def test_load_and_preprocess1(self):
        corpus_file = "Data/dogs-train.txt"
        expected = [["i", "like", "dogs"],
                    ["dogs", "like", "walks"],
                    ["she", "walks", "her", "dogs", "and", "cats"],
                    ["her", "dogs", "like", "cats"]]
        self.unigram.load_and_preprocess(corpus_file, train=True)
        actual = self.unigram.training_data
        self.assertListEqual(expected, actual)

    def test_load_and_preprocess2(self):
        corpus_file = "Data/dogs-test.txt"
        self.unigram.vocab = [
            'and', 'cats', 'dogs', 'her', 'i', 'like', 'she', 'walks'
        ]
        expected = [
            ['cats', 'like', 'dogs'],
            ['dogs', 'like', '<UNK>', 'cats']
        ]
        self.unigram.load_and_preprocess(corpus_file, train=False)
        actual = self.unigram.test_data
        self.assertListEqual(expected, actual)

    def test_count_unigrams(self):
        self.unigram.training_data = [
            ["i", "like", "dogs"],
            ["dogs", "like", "walks"],
            ["she", "walks", "her", "dogs", "and", "cats"],
            ["her", "dogs", "like", "cats"]
        ]
        expected = {
            "and": 1,
            "cats": 2,
            "dogs": 4,
            "her": 2,
            "i": 1,
            "like": 3,
            "she": 1,
            "walks": 2
        }
        self.unigram.count_unigrams()
        actual = self.unigram.counts
        self.assertDictEqual(expected, actual)

    def test_remove_low_freq_words(self):
        self.unigram.counts = {
            "and": 1,
            "cats": 2,
            "dogs": 4,
            "her": 2,
            "i": 1,
            "like": 3,
            "she": 1,
            "walks": 2
        }
        expected_counts = {
            UNK: 3,
            "cats": 2,
            "dogs": 4,
            "her": 2,
            "like": 3,
            "walks": 2
        }
        self.unigram.remove_low_freq_words()
        actual_counts = self.unigram.counts
        self.assertDictEqual(expected_counts, actual_counts)

    def test_set_vocab(self):
        self.unigram.counts = {
            UNK: 3,
            "cats": 2,
            "dogs": 4,
            "her": 2,
            "like": 3,
            "walks": 2
        }
        expected_vocab = [UNK, "cats", "dogs", "her", "like", "walks"]
        self.unigram.set_vocab()
        actual_vocab = self.unigram.vocab
        self.assertListEqual(expected_vocab, actual_vocab, msg=f"\nvocab should be sorted.")

    def test_set_col_idx_map(self):
        self.unigram.vocab = [
            UNK, "cats", "dogs", "her", "like", "walks"
        ]
        expected = {
            UNK: 0,
            "cats": 1,
            "dogs": 2,
            "her": 3,
            "like": 4,
            "walks": 5
        }
        self.unigram.set_col_idx_map()
        actual = self.unigram.col_idx_map
        self.assertDictEqual(expected, actual)

    def test_calc_probs1(self):
        self.unigram.counts = {
            UNK: 4,
            "cats": 3,
            "dogs": 5,
            "her": 3,
            "like": 4,
            "walks": 3
        }
        self.unigram.vocab = [
            UNK, "cats", "dogs", "her", "like", "walks"
        ]
        self.unigram.col_idx_map = {
            UNK: 0,
            "cats": 1,
            "dogs": 2,
            "her": 3,
            "like": 4,
            "walks": 5
        }
        self.unigram.calc_probs()
        sum_probs = np.sum(self.unigram.probs)
        self.assertAlmostEqual(1, sum_probs, delta=.000001)

    def test_calc_probs2(self):
        self.unigram.counts = {
            UNK: 4,
            "cats": 3,
            "dogs": 5,
            "her": 3,
            "like": 4,
            "walks": 3
        }
        self.unigram.vocab = [
            UNK, "cats", "dogs", "her", "like", "walks"
        ]
        self.unigram.col_idx_map = {
            UNK: 0,
            "cats": 1,
            "dogs": 2,
            "her": 3,
            "like": 4,
            "walks": 5
        }
        expected = np.array([
            0.18181818, 0.13636364, 0.22727273, 0.13636364, 0.18181818, 0.13636364
        ])
        self.unigram.calc_probs()
        actual = self.unigram.probs
        self.assertTrue(np.allclose(expected, actual, atol=.00001),
                        msg=f"\nexpected:\n{expected}\nbut got:\n{actual}")

    def test_calc_log10_probs(self):
        self.unigram.probs = np.array([
            0.18181818, 0.13636364, 0.22727273, 0.13636364, 0.18181818, 0.13636364
        ])
        expected = np.array([
            -0.74036269, -0.86530143, -0.64345268, -0.86530143, -0.74036269, -0.86530143
        ])
        self.unigram.calc_log10_probs()
        actual = self.unigram.logprobs
        self.assertTrue(np.allclose(expected, actual, atol=.001),
                        msg=f"\nexpected:\n{expected}\nbut got:\n{actual}")

    def test_train1(self):
        self.unigram.train("Data/dogs-train.txt")
        expected_counts = {
            UNK: 3,
            "cats": 2,
            "dogs": 4,
            "her": 2,
            "like": 3,
            "walks": 2
        }
        actual_counts = self.unigram.counts
        self.assertDictEqual(expected_counts, actual_counts)

    def test_train2(self):
        self.unigram.train("Data/dogs-train.txt")
        expected_vocab = [UNK, "cats", "dogs", "her", "like", "walks"]
        actual_vocab = self.unigram.vocab
        self.assertListEqual(expected_vocab, actual_vocab)

    def test_train3(self):
        self.unigram.train("Data/dogs-train.txt")
        expected_col_map = {
            UNK: 0,
            "cats": 1,
            "dogs": 2,
            "her": 3,
            "like": 4,
            "walks": 5
        }
        actual_col_map = self.unigram.col_idx_map
        self.assertDictEqual(expected_col_map, actual_col_map)

    def test_train4(self):
        self.unigram.train("Data/dogs-train.txt")
        expected_probs = np.array([
            0.1875, 0.125, 0.25, 0.125, 0.1875, 0.125
        ])
        actual_probs = self.unigram.probs
        self.assertTrue(np.allclose(expected_probs, actual_probs, atol=.0001),
                        msg=f"\nexpected:\n{expected_probs}\nbut got:\n{actual_probs}")

    def test_train5(self):
        self.unigram.train("Data/dogs-train.txt")
        expected_logprobs = np.array([
            -0.72699873, -0.90308999, -0.60205999, -0.90308999, -0.72699873, -0.90308999
        ])
        actual_logprobs = self.unigram.logprobs
        self.assertTrue(np.allclose(expected_logprobs, actual_logprobs, atol=.0001),
                        msg=f"\nexpected:\n{expected_logprobs}\nbut got:\n{actual_logprobs}")

    def test_generate_sent(self):
        self.unigram.vocab = [
            UNK, "cats", "dogs", "her", "like", "walks"
        ]
        self.unigram.probs = np.array([
            0.1875, 0.125, 0.25, 0.125, 0.1875, 0.125
        ])
        self.unigram.col_idx_map = {
            UNK: 0,
            "cats": 1,
            "dogs": 2,
            "her": 3,
            "like": 4,
            "walks": 5
        }
        sent_len = 10
        sent = self.unigram.generate_sentence(sent_len)
        actual = len(sent)
        self.assertEqual(sent_len, actual, msg=f"expected sentence of length {sent_len}, but got {len(sent)}")

    def test_calc_sent_logprob1(self):
        self.unigram.col_idx_map = {
            UNK: 0,
            "cats": 1,
            "dogs": 2,
            "her": 3,
            "like": 4,
            "walks": 5
        }
        self.unigram.logprobs = np.array([
            -0.72699873, -0.90308999, -0.60205999, -0.90308999, -0.72699873, -0.90308999
        ])
        sent = ["cats", "like", "dogs"]
        expected = -2.23215
        actual = self.unigram.calc_sent_logprob(sent)
        self.assertAlmostEqual(expected, actual, delta=.001)

    def test_calc_sent_logprob2(self):
        self.unigram.col_idx_map = {
            UNK: 0,
            "cats": 1,
            "dogs": 2,
            "her": 3,
            "like": 4,
            "walks": 5
        }
        self.unigram.logprobs = np.array([
            -0.72699873, -0.90308999, -0.60205999, -0.90308999, -0.72699873, -0.90308999
        ])
        sent = ["cats", "like", UNK, "dogs"]
        expected = -2.959147
        actual = self.unigram.calc_sent_logprob(sent)
        self.assertAlmostEqual(expected, actual, delta=.001)

    def test_calc_perplexity1(self):
        self.unigram.col_idx_map = {
            UNK: 0,
            "cats": 1,
            "dogs": 2,
            "her": 3,
            "like": 4,
            "walks": 5
        }
        self.unigram.logprobs = np.array([
            -0.74036269, -0.86530143, -0.64345268, -0.86530143, -0.74036269, -0.86530143
        ])
        test_sentences = [
            ["cats", "like", "dogs"]
        ]
        expected = 5.6196
        actual = self.unigram.calc_perplexity(test_sentences)
        self.assertAlmostEqual(expected, actual, delta=.001)

    def test_calc_perplexity2(self):
        self.unigram.col_idx_map = {
            UNK: 0,
            "cats": 1,
            "dogs": 2,
            "her": 3,
            "like": 4,
            "walks": 5
        }
        self.unigram.logprobs = np.array([
            -0.74036269, -0.86530143, -0.64345268, -0.86530143, -0.74036269, -0.86530143
        ])
        test_sentences = [
            ["dogs", "like", UNK, "cats"]
        ]
        expected = 5.5895
        actual = self.unigram.calc_perplexity(test_sentences)
        self.assertAlmostEqual(expected, actual, delta=.001)

    def test_calc_perplexity3(self):
        self.unigram.col_idx_map = {
            UNK: 0,
            "cats": 1,
            "dogs": 2,
            "her": 3,
            "like": 4,
            "walks": 5
        }
        self.unigram.logprobs = np.array([
            -0.74036269, -0.86530143, -0.64345268, -0.86530143, -0.74036269, -0.86530143
        ])
        test_sentences = [
            ["cats", "like", "dogs"],
            ["dogs", "like", UNK, "cats"]
        ]
        expected = 5.6023
        actual = self.unigram.calc_perplexity(test_sentences)
        self.assertAlmostEqual(expected, actual, delta=.001)

    def test_test(self):
        self.unigram.vocab = [
            UNK, "cats", "dogs", "her", "like", "walks"
        ]
        self.unigram.col_idx_map = {
            UNK: 0,
            "cats": 1,
            "dogs": 2,
            "her": 3,
            "like": 4,
            "walks": 5
        }
        self.unigram.logprobs = np.array([
            -0.74036269, -0.86530143, -0.64345268, -0.86530143, -0.74036269, -0.86530143
        ])
        expected = 5.6023
        actual = self.unigram.test("Data/dogs-test.txt")
        self.assertAlmostEqual(expected, actual, delta=.001)


if __name__ == '__main__':
    unittest.main()
