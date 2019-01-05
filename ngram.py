import os
import math


class Ngram:
    def __init__(self):
        self.SOS = '<s>'
        self.EOS = '</s>'

    def __load_data(self, fname):
        if not os.path.isfile(fname):
            print('{} does not exist!'.format(fname))
            return

        with open(fname, 'r') as f:
            lines = [[self.SOS] + l.strip().split(' ' ) + [self.EOS] for l in f.readlines()]
            flat_lines = [w for l in lines for w in l]
            n_tokens = len(flat_lines) - 2 * len(lines) # remove SOS and EOS counts
            vocabs = sorted(list(set(flat_lines)))
            unigram_freq = {w: flat_lines.count(w) for w in vocabs}

            bigrams = []
            for l in lines:
                for i in range(0, len(l) - 1):
                   bigrams.append(' '.join(l[i:i+2]))
            uniq_bigrams = sorted(list(set(bigrams)))
            bigram_freq = {b: bigrams.count(b) for b in uniq_bigrams}

            print('bigrams:', uniq_bigrams)
            print('bigram freq:', bigram_freq)

            return {
                'lines'       : lines,
                'n_tokens'    : n_tokens,
                'vocabs'      : vocabs,
                'uniq_bigrams': uniq_bigrams,
                'unigram_freq': unigram_freq,
                'bigram_freq' : bigram_freq
            }

        print('Failed to read file: {}'.format(fname))

    def train(self, fname, n=2):
        data = self.__load_data(fname)
        if data is None:
            return
        self.train_data = data

        vocabs = data['vocabs']
        unigram_freq = data['unigram_freq']
        bigram_freq = data['bigram_freq']
        print('vocabs:', vocabs)
        print('unigram_freq:', unigram_freq)

        print('Training results:----------')
        for big in bigram_freq.keys():
            toks = big.split(' ')
            p_w = bigram_freq[big] / unigram_freq[toks[0]]
            print('P({})={}'.format(big, p_w))

    def dump(self, fname, n=2):
        lines = []
        if n == 1:
            for uni in self.train_data['unigram_freq']:
                p_w = self.train_data['unigram_freq'][uni] / self.train_data['n_tokens']
                lines.append('\t'.join([uni, str(p_w) + '\n']))
        elif n == 2:
            for big in self.train_data['bigram_freq'].keys():
                toks = big.split(' ')
                p_w = self.train_data['bigram_freq'][big] / self.train_data['unigram_freq'][toks[0]]
                lines.append('\t'.join([big, str(p_w) + '\n']))
        else:
            print('Unknown n-gram:', n)

        with open(fname, 'w') as f:
            print('save file: {}'.format(fname))
            f.writelines(lines)

    def test(self, test_fname, lambda1=0.3, lambda2=0.3, V=1e6):
        test_data = self.__load_data(test_fname)
        if test_data is None:
            print('test data is None')
            return
        train_data = self.train_data
        if train_data is None:
            print('train data is None')
            return

        entropy = 0
        log_likelihood = 0
        log2_likelihood = 0
        for l in test_data['lines']:
            p_sent = 1.
            for i in range(0, len(l) - 1):
                w_prev = l[i]
                w = l[i+1]
                bi_token = ' '.join(l[i:i+2])

                if w in train_data['vocabs']:
                    p_ml = train_data['unigram_freq'][w] / train_data['n_tokens']
                else:
                    p_ml = 0
                p_uni = lambda1 * p_ml + (1 - lambda1) * (1.0 / V)

                if bi_token in train_data['bigram_freq']:
                    p_bi = train_data['bigram_freq'][bi_token] / train_data['unigram_freq'][w_prev]
                else:
                    p_bi = 0

                p_bi = lambda2 * p_bi + (1 - lambda2) * p_uni
                p_sent *= p_bi

            log_likelihood += math.log(p_sent)
            log2_likelihood += (-1) * math.log2(p_sent)

        self.log_likelihood = log2_likelihood
        entropy = log2_likelihood / (test_data['n_tokens'])
        self.entropy = entropy
        ppl = math.pow(2, entropy)
        self.perplexity = ppl

        # calculate coverage
        test_flat_lines = [w for l in test_data['lines'] for w in l]
        contains_count = 0
        for w in test_flat_lines:
            if w in train_data['vocabs']:
                contains_count += 1
        coverage = contains_count / len(test_flat_lines)
        self.coverage = coverage

        print('log-likelihood:', log_likelihood)
        print('entropy:', entropy)
        print('perplexity:', ppl)
        print('coverage:', coverage)

    def clear(self):
        self.train_data      = None
        self.test_data       = None
        self.log2_likelihood = None
        self.entropy         = None
        self.ppl             = None
        self.coverage        = None
