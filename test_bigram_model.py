import unittest

from bigram_model import *


class BigramTestCase(unittest.TestCase):

    @classmethod
    def setUp(self):
        self.bigram = Bigram()
    @classmethod
    def tearDown(self):
        self.bigram = None

    def test_preprocess_train_sentence(self):
        self.bigram.vocab = [UNK, "this", "is", "it"]
        sent = ["this", "is", "easy", "!"]
        expected = [BOS, "this", "is", UNK, UNK, EOS]
        actual = self.bigram.preprocess_train_sentence(sent)
        self.assertListEqual(expected, actual)

    def test_preprocess_test_sentence(self):
        self.bigram.vocab = [UNK, "this", "is", "it"]
        sent = "this is easy !"
        expected = [BOS, "this", "is", UNK, UNK, EOS]
        actual = self.bigram.preprocess_test_sentence(sent)
        self.assertListEqual(expected, actual)

    def test_load_and_preprocess1(self):
        self.bigram.vocab = [UNK, 'cats', 'dogs', 'her', 'like', 'walks']
        sentences = [
            [UNK, "like", "dogs"],
            ["dogs", "like", "walks"],
            [UNK, "walks", "her", "dogs", UNK, "cats"],
            ["her", "dogs", "like", "cats"]
        ]
        expected = [
            [BOS, UNK, "like", "dogs", EOS],
            [BOS, "dogs", "like", "walks", EOS],
            [BOS, UNK, "walks", "her", "dogs", UNK, "cats", EOS],
            [BOS, "her", "dogs", "like", "cats", EOS]
        ]
        self.bigram.unigram_model.training_data = sentences
        self.bigram.load_and_preprocess(train=True)
        actual = self.bigram.training_data
        self.assertListEqual(expected, actual)

    def test_load_and_preprocess2(self):
        corpus_file = "Data/dogs-test.txt"
        self.bigram.vocab = [UNK, 'cats', 'dogs', 'her', 'like', 'walks']
        expected = [
            [BOS, "cats", "like", "dogs", EOS],
            [BOS, "dogs", "like", UNK, "cats", EOS]
        ]
        self.bigram.load_and_preprocess(train=False, corpus_file=corpus_file)
        actual = self.bigram.test_data
        self.assertListEqual(expected, actual)

    def test_set_vocab(self):
        self.bigram.unigram_model.vocab = [
            UNK, 'cats', 'dogs', 'her', 'like', 'walks'
        ]
        expected = [
            UNK, 'cats', 'dogs', 'her', 'like', 'walks'
        ]
        self.bigram.set_vocab()
        actual = self.bigram.vocab
        self.assertListEqual(expected, actual)

    def test_set_col_row_idx_maps1(self):
        self.bigram.vocab = [UNK, 'cats', 'dogs', 'her', 'like', 'walks']
        expected_row_map = {
            UNK: 0,
            'cats': 1,
            'dogs': 2,
            'her': 3,
            'like': 4,
            'walks': 5,
            BOS: 6
        }
        self.bigram.set_col_row_idx_maps()
        actual_row_map = self.bigram.row_idx_map
        self.assertDictEqual(expected_row_map, actual_row_map)

    def test_set_col_row_idx_maps2(self):
        self.bigram.vocab = [UNK, 'cats', 'dogs', 'her', 'like', 'walks']
        expected_col_map = {
            UNK: 0,
            'cats': 1,
            'dogs': 2,
            'her': 3,
            'like': 4,
            'walks': 5,
            EOS: 6
        }
        self.bigram.set_col_row_idx_maps()
        actual_col_map = self.bigram.col_idx_map
        self.assertDictEqual(expected_col_map, actual_col_map)

    def test_count_bigrams(self):
        self.bigram.training_data = [
            [BOS, UNK, "like", "dogs", EOS],
            [BOS, "dogs", "like", "walks", EOS],
            [BOS, UNK, "walks", "her", "dogs", UNK, "cats", EOS],
            [BOS, "her", "dogs", "like", "cats", EOS]
        ]
        self.bigram.row_idx_map = {
            UNK: 0,
            'cats': 1,
            'dogs': 2,
            'her': 3,
            'like': 4,
            'walks': 5,
            BOS: 6
        }
        self.bigram.col_idx_map = {
            UNK: 0,
            'cats': 1,
            'dogs': 2,
            'her': 3,
            'like': 4,
            'walks': 5,
            EOS: 6
        }
        expected_matrix = np.array([
            [0, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 2],
            [1, 0, 0, 0, 2, 0, 1],
            [0, 0, 2, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [2, 0, 1, 1, 0, 0, 0]
        ])
        self.bigram.count_bigrams()
        bigram_matrix = self.bigram.counts
        self.assertTrue(np.allclose(expected_matrix, bigram_matrix, rtol=.001),
                        msg=f"\nexpected:\n{expected_matrix}\nbut got:\n{bigram_matrix}")

    def test_smooth1(self):
        self.bigram.counts = np.array([
            [0, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 2],
            [1, 0, 0, 0, 2, 0, 1],
            [0, 0, 2, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [2, 0, 1, 1, 0, 0, 0]
        ])
        expected = np.array([
            [1, 2, 1, 1, 2, 2, 1],
            [1, 1, 1, 1, 1, 1, 3],
            [2, 1, 1, 1, 3, 1, 2],
            [1, 1, 3, 1, 1, 1, 1],
            [1, 2, 2, 1, 1, 2, 1],
            [1, 1, 1, 2, 1, 1, 2],
            [3, 1, 2, 2, 1, 1, 1]
        ])
        self.bigram.smooth(k=1)
        actual = self.bigram.counts
        self.assertTrue(np.allclose(expected, actual, rtol=.001),
                        msg=f"\nexpected:\n{expected}\nbut got:\n{actual}")

    def test_smooth2(self):
        self.bigram.counts = np.array([
            [0, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 2],
            [1, 0, 0, 0, 2, 0, 1],
            [0, 0, 2, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [2, 0, 1, 1, 0, 0, 0]
        ])
        expected = np.array([
            [0.5, 1.5, 0.5, 0.5, 1.5, 1.5, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.5],
            [1.5, 0.5, 0.5, 0.5, 2.5, 0.5, 1.5],
            [0.5, 0.5, 2.5, 0.5, 0.5, 0.5, 0.5],
            [0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 0.5],
            [0.5, 0.5, 0.5, 1.5, 0.5, 0.5, 1.5],
            [2.5, 0.5, 1.5, 1.5, 0.5, 0.5, 0.5]
        ])
        self.bigram.smooth(k=0.5)
        actual = self.bigram.counts
        self.assertTrue(np.allclose(expected, actual),
                        msg=f"\nexpected:\n{expected}\nbut got:\n{actual}")

    def test_calc_probs(self):
        self.bigram.counts = np.array([
            [1, 2, 1, 1, 2, 2, 1],
            [1, 1, 1, 1, 1, 1, 3],
            [2, 1, 1, 1, 3, 1, 2],
            [1, 1, 3, 1, 1, 1, 1],
            [1, 2, 2, 1, 1, 2, 1],
            [1, 1, 1, 2, 1, 1, 2],
            [3, 1, 2, 2, 1, 1, 1]
        ])
        expected = np.array([
            [0.1       , 0.2       , 0.1       , 0.1       , 0.2       , 0.2       ,  0.1       ],
            [0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111,  0.33333333],
            [0.18181818, 0.09090909, 0.09090909, 0.09090909, 0.27272727, 0.09090909,  0.18181818],
            [0.11111111, 0.11111111, 0.33333333, 0.11111111, 0.11111111, 0.11111111,  0.11111111],
            [0.1       , 0.2       , 0.2       , 0.1       , 0.1       , 0.2       ,  0.1       ],
            [0.11111111, 0.11111111, 0.11111111, 0.22222222, 0.11111111, 0.11111111,  0.22222222],
            [0.27272727, 0.09090909, 0.18181818, 0.18181818, 0.09090909, 0.09090909,  0.09090909]
        ])
        self.bigram.calc_probs()
        actual = self.bigram.probs
        self.assertTrue(np.allclose(expected, actual),
                        msg=f"\nexpected:\n{expected}\nbut got:\n{actual}")

    def test_calc_log10_probs(self):
        self.bigram.probs = np.array([
            [0.1       , 0.2       , 0.1       , 0.1       , 0.2       , 0.2       ,  0.1       ],
            [0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111,  0.33333333],
            [0.18181818, 0.09090909, 0.09090909, 0.09090909, 0.27272727, 0.09090909,  0.18181818],
            [0.11111111, 0.11111111, 0.33333333, 0.11111111, 0.11111111, 0.11111111,  0.11111111],
            [0.1       , 0.2       , 0.2       , 0.1       , 0.1       , 0.2       ,  0.1       ],
            [0.11111111, 0.11111111, 0.11111111, 0.22222222, 0.11111111, 0.11111111,  0.22222222],
            [0.27272727, 0.09090909, 0.18181818, 0.18181818, 0.09090909, 0.09090909,  0.09090909]
        ])
        expected = np.array([
            [-1.        , -0.69897   , -1.        , -1.        , -0.69897   , -0.69897   , -1.        ],
            [-0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.47712126],
            [-0.74036269, -1.04139269, -1.04139269, -1.04139269, -0.56427143, -1.04139269, -0.74036269],
            [-0.95424251, -0.95424251, -0.47712126, -0.95424251, -0.95424251, -0.95424251, -0.95424251],
            [-1.        , -0.69897   , -0.69897   , -1.        , -1.        ,  -0.69897  , -1.        ],
            [-0.95424251, -0.95424251, -0.95424251, -0.65321252, -0.95424251, -0.95424251, -0.65321252],
            [-0.56427143, -1.04139269, -0.74036269, -0.74036269, -1.04139269, -1.04139269, -1.04139269]
        ])
        self.bigram.calc_log10_probs()
        actual = self.bigram.logprobs
        self.assertTrue(np.allclose(expected, actual, atol=.00001),
                        msg=f"\nexpected:\n{expected}\nbut got:\n{actual}")

    def test_train(self):
        expected_logprobs = np.array([
            [-1.        , -0.69897   , -1.        , -1.        , -0.69897   , -0.69897   , -1.        ],
            [-0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.47712126],
            [-0.74036269, -1.04139269, -1.04139269, -1.04139269, -0.56427143, -1.04139269, -0.74036269],
            [-0.95424251, -0.95424251, -0.47712126, -0.95424251, -0.95424251, -0.95424251, -0.95424251],
            [-1.        , -0.69897   , -0.69897   , -1.        , -1.        ,  -0.69897  , -1.        ],
            [-0.95424251, -0.95424251, -0.95424251, -0.65321252, -0.95424251, -0.95424251, -0.65321252],
            [-0.56427143, -1.04139269, -0.74036269, -0.74036269, -1.04139269, -1.04139269, -1.04139269]
        ])
        self.bigram.train("Data/dogs-train.txt", k=1)
        actual_logprobs = self.bigram.logprobs
        self.assertTrue(np.allclose(expected_logprobs, actual_logprobs, atol=.00001),
                        msg=f"\nexpected:\n{expected_logprobs}\nbut got:\n{actual_logprobs}")

    def test_generate_sentence(self):
        self.bigram.row_idx_map = {
            UNK: 0,
            'cats': 1,
            'dogs': 2,
            'her': 3,
            'like': 4,
            'walks': 5,
            BOS: 6
        }
        self.bigram.col_idx_map = {
            UNK: 0,
            'cats': 1,
            'dogs': 2,
            'her': 3,
            'like': 4,
            'walks': 5,
            EOS: 6
        }
        self.bigram.probs = np.array([
            [0.1, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1],
            [0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.33333333],
            [0.18181818, 0.09090909, 0.09090909, 0.09090909, 0.27272727, 0.09090909, 0.18181818],
            [0.11111111, 0.11111111, 0.33333333, 0.11111111, 0.11111111, 0.11111111, 0.11111111],
            [0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.1],
            [0.11111111, 0.11111111, 0.11111111, 0.22222222, 0.11111111, 0.11111111, 0.22222222],
            [0.27272727, 0.09090909, 0.18181818, 0.18181818, 0.09090909, 0.09090909, 0.09090909]
        ])

        for i in range(10):
            sentence = self.bigram.generate_sentence()
            self.assertEqual(BOS, sentence[0],
                             msg="Sentences should start with BOS.")
            self.assertEqual(EOS, sentence[len(sentence) - 1],
                             msg="Sentences should end with EOS.")

    def test_calc_sent_logprob(self):
        self.bigram.row_idx_map = {
            UNK: 0,
            'cats': 1,
            'dogs': 2,
            'her': 3,
            'like': 4,
            'walks': 5,
            BOS: 6
        }
        self.bigram.col_idx_map = {
            UNK: 0,
            'cats': 1,
            'dogs': 2,
            'her': 3,
            'like': 4,
            'walks': 5,
            EOS: 6
        }
        self.bigram.logprobs = np.array([
            [-1.        , -0.69897   , -1.        , -1.        , -0.69897   , -0.69897   , -1.        ],
            [-0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.47712126],
            [-0.74036269, -1.04139269, -1.04139269, -1.04139269, -0.56427143, -1.04139269, -0.74036269],
            [-0.95424251, -0.95424251, -0.47712126, -0.95424251, -0.95424251, -0.95424251, -0.95424251],
            [-1.        , -0.69897   , -0.69897   , -1.        , -1.        ,  -0.69897  , -1.        ],
            [-0.95424251, -0.95424251, -0.95424251, -0.65321252, -0.95424251, -0.95424251, -0.65321252],
            [-0.56427143, -1.04139269, -0.74036269, -0.74036269, -1.04139269, -1.04139269, -1.04139269]
        ])
        sent = [BOS, "cats", "like", "dogs", EOS]
        expected = -3.43496789
        actual = self.bigram.calc_sent_logprob(sent)
        self.assertAlmostEqual(expected, actual, delta=.001)

    def test_calc_perplexity1(self):
        self.bigram.row_idx_map = {
            UNK: 0,
            'cats': 1,
            'dogs': 2,
            'her': 3,
            'like': 4,
            'walks': 5,
            BOS: 6
        }
        self.bigram.col_idx_map = {
            UNK: 0,
            'cats': 1,
            'dogs': 2,
            'her': 3,
            'like': 4,
            'walks': 5,
            EOS: 6
        }
        self.bigram.logprobs = np.array([
            [-1., -0.69897, -1., -1., -0.69897, -0.69897, -1.],
            [-0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.47712126],
            [-0.74036269, -1.04139269, -1.04139269, -1.04139269, -0.56427143, -1.04139269, -0.74036269],
            [-0.95424251, -0.95424251, -0.47712126, -0.95424251, -0.95424251, -0.95424251, -0.95424251],
            [-1., -0.69897, -0.69897, -1., -1., -0.69897, -1.],
            [-0.95424251, -0.95424251, -0.95424251, -0.65321252, -0.95424251, -0.95424251, -0.65321252],
            [-0.56427143, -1.04139269, -0.74036269, -0.74036269, -1.04139269, -1.04139269, -1.04139269]
        ])
        test_sentences = [
            [BOS, "cats", "like", "dogs", EOS]
        ]
        expected = 7.223405117
        actual = self.bigram.calc_perplexity(test_sentences)
        self.assertAlmostEqual(expected, actual, delta=.001)

    def test_get_perplexity2(self):
        self.bigram.row_idx_map = {
            UNK: 0,
            'cats': 1,
            'dogs': 2,
            'her': 3,
            'like': 4,
            'walks': 5,
            BOS: 6
        }
        self.bigram.col_idx_map = {
            UNK: 0,
            'cats': 1,
            'dogs': 2,
            'her': 3,
            'like': 4,
            'walks': 5,
            EOS: 6
        }
        self.bigram.logprobs = np.array([
            [-1., -0.69897, -1., -1., -0.69897, -0.69897, -1.],
            [-0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.47712126],
            [-0.74036269, -1.04139269, -1.04139269, -1.04139269, -0.56427143, -1.04139269, -0.74036269],
            [-0.95424251, -0.95424251, -0.47712126, -0.95424251, -0.95424251, -0.95424251, -0.95424251],
            [-1., -0.69897, -0.69897, -1., -1., -0.69897, -1.],
            [-0.95424251, -0.95424251, -0.95424251, -0.65321252, -0.95424251, -0.95424251, -0.65321252],
            [-0.56427143, -1.04139269, -0.74036269, -0.74036269, -1.04139269, -1.04139269, -1.04139269]
        ])
        test_sentences = [
            [BOS, "dogs", "like", UNK, "cats", EOS]
        ]
        expected = 4.9675823
        actual = self.bigram.calc_perplexity(test_sentences)
        self.assertAlmostEqual(expected, actual, delta=.001)

    def test_get_perplexity3(self):
        self.bigram.row_idx_map = {
            UNK: 0,
            'cats': 1,
            'dogs': 2,
            'her': 3,
            'like': 4,
            'walks': 5,
            BOS: 6
        }
        self.bigram.col_idx_map = {
            UNK: 0,
            'cats': 1,
            'dogs': 2,
            'her': 3,
            'like': 4,
            'walks': 5,
            EOS: 6
        }
        self.bigram.logprobs = np.array([
            [-1., -0.69897, -1., -1., -0.69897, -0.69897, -1.],
            [-0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.47712126],
            [-0.74036269, -1.04139269, -1.04139269, -1.04139269, -0.56427143, -1.04139269, -0.74036269],
            [-0.95424251, -0.95424251, -0.47712126, -0.95424251, -0.95424251, -0.95424251, -0.95424251],
            [-1., -0.69897, -0.69897, -1., -1., -0.69897, -1.],
            [-0.95424251, -0.95424251, -0.95424251, -0.65321252, -0.95424251, -0.95424251, -0.65321252],
            [-0.56427143, -1.04139269, -0.74036269, -0.74036269, -1.04139269, -1.04139269, -1.04139269]
        ])
        test_sentences = [
            [BOS, "cats", "like", "dogs", EOS],
            [BOS, "dogs", "like", UNK, "cats", EOS]
        ]
        expected = 5.8669226
        actual = self.bigram.calc_perplexity(test_sentences)
        self.assertAlmostEqual(expected, actual, delta=.001)

    def test_test1(self):
        self.bigram.unigram_model.vocab = [UNK, 'cats', 'dogs', 'her', 'like', 'walks']
        self.bigram.unigram_model.col_idx_map = {
            UNK: 0,
            'cats': 1,
            'dogs': 2,
            'her': 3,
            'like': 4,
            'walks': 5
        }
        self.bigram.unigram_model.logprobs = np.array([
            -0.74036269, -0.86530143, -0.64345268, -0.86530143, -0.74036269, -0.86530143
        ])
        self.bigram.vocab = [UNK, 'cats', 'dogs', 'her', 'like', 'walks']
        self.bigram.row_idx_map = {
            UNK: 0,
            'cats': 1,
            'dogs': 2,
            'her': 3,
            'like': 4,
            'walks': 5,
            BOS: 6
        }
        self.bigram.col_idx_map = {
            UNK: 0,
            'cats': 1,
            'dogs': 2,
            'her': 3,
            'like': 4,
            'walks': 5,
            EOS: 6
        }
        self.bigram.logprobs = np.array([
            [-1., -0.69897, -1., -1., -0.69897, -0.69897, -1.],
            [-0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.47712126],
            [-0.74036269, -1.04139269, -1.04139269, -1.04139269, -0.56427143, -1.04139269, -0.74036269],
            [-0.95424251, -0.95424251, -0.47712126, -0.95424251, -0.95424251, -0.95424251, -0.95424251],
            [-1., -0.69897, -0.69897, -1., -1., -0.69897, -1.],
            [-0.95424251, -0.95424251, -0.95424251, -0.65321252, -0.95424251, -0.95424251, -0.65321252],
            [-0.56427143, -1.04139269, -0.74036269, -0.74036269, -1.04139269, -1.04139269, -1.04139269]
        ])

        expected = 5.6023585
        pp_unigram, _ = self.bigram.test("Data/dogs-test.txt")
        self.assertAlmostEqual(expected, pp_unigram, delta=.001)

    def test_test2(self):
        self.bigram.unigram_model.vocab = [UNK, 'cats', 'dogs', 'her', 'like', 'walks']
        self.bigram.unigram_model.col_idx_map = {
            UNK: 0,
            'cats': 1,
            'dogs': 2,
            'her': 3,
            'like': 4,
            'walks': 5
        }
        self.bigram.unigram_model.logprobs = np.array([
            -0.74036269, -0.86530143, -0.64345268, -0.86530143, -0.74036269, -0.86530143
        ])
        self.bigram.vocab = [UNK, 'cats', 'dogs', 'her', 'like', 'walks']
        self.bigram.row_idx_map = {
            UNK: 0,
            'cats': 1,
            'dogs': 2,
            'her': 3,
            'like': 4,
            'walks': 5,
            BOS: 6
        }
        self.bigram.col_idx_map = {
            UNK: 0,
            'cats': 1,
            'dogs': 2,
            'her': 3,
            'like': 4,
            'walks': 5,
            EOS: 6
        }
        self.bigram.logprobs = np.array([
            [-1., -0.69897, -1., -1., -0.69897, -0.69897, -1.],
            [-0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.95424251, -0.47712126],
            [-0.74036269, -1.04139269, -1.04139269, -1.04139269, -0.56427143, -1.04139269, -0.74036269],
            [-0.95424251, -0.95424251, -0.47712126, -0.95424251, -0.95424251, -0.95424251, -0.95424251],
            [-1., -0.69897, -0.69897, -1., -1., -0.69897, -1.],
            [-0.95424251, -0.95424251, -0.95424251, -0.65321252, -0.95424251, -0.95424251, -0.65321252],
            [-0.56427143, -1.04139269, -0.74036269, -0.74036269, -1.04139269, -1.04139269, -1.04139269]
        ])

        expected = 5.8669226
        _, pp_bigram = self.bigram.test("Data/dogs-test.txt")
        self.assertAlmostEqual(expected, pp_bigram, delta=.001)


if __name__ == '__main__':
    unittest.main()
