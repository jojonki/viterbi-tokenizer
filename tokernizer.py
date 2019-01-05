import math
from ngram import Ngram
# Use Graham's example.
# http://www.phontron.com/slides/nlp-programming-ja-03-ws.pdf

INF = 1e10

class Tokenizer:
    def __init__(self):
        pass

    def set_dictionary(self, probas):
        self.probas = probas

    def forward(self, sent, probas=None):
        if probas is not None:
            self.probas = probas
        if self.probas is None:
            print('You need to set ngram probabilities')
            return
        probas = self.probas

        L = len(sent)
        node_list = [{'best_score': None, 'best_edge': None} for _ in range(1 + L)] # includes first None node
        node_list[0]['best_score'] = 0

        # forward
        for end in range(1, L + 1):
            node_list[end]['best_score'] = INF
            for beg in range(0, end):
                token = sent[beg:end]
                if token in probas.keys():
                    prob = probas[token]
                    score = node_list[beg]['best_score'] + (-1) * math.log(prob)
                    if score < node_list[end]['best_score']:
                        node_list[end]['best_score'] = score
                        node_list[end]['best_edge'] = (beg, end)
                else:
                    # print('UNK:', token)
                    # skip UNK words
                    pass

        # print('node_list:', node_list)
        return node_list

    def backward(self, sent, node_list):
        words = []
        next_edge = node_list[-1]['best_edge']
        while next_edge is not None:
            words.append(sent[next_edge[0]:next_edge[1]])
            next_edge = node_list[next_edge[0]]['best_edge']
        words = words[::-1]

        return words

    def tokenize(self, sent):
        if self.probas is None:
            print('You need to set ngram probabilities')
            return

        probas = self.probas
        node_list = self.forward(sent, probas)
        return self.backward(sent, node_list)
