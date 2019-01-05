import argparse
from tokernizer import Tokenizer
# Use Graham's example.
# http://www.phontron.com/slides/nlp-programming-ja-03-ws.pdf

INF = 1e10

parser = argparse.ArgumentParser()
parser.add_argument('--test_file', type=str, metavar='PATH', help='test file path')
parser.add_argument('--lm', type=str, metavar='PATH', help='trained language model')
parser.add_argument('--dump_file', type=str, metavar='PATH', default='my_answer.word', help='tokenized result file')
args = parser.parse_args()


def load_model(fpath):
    probas = {}
    with open(fpath, 'r') as f:
        lines = f.readlines()
        for l in lines:
            token, prob = l.rstrip().split('\t')
            probas[token] = float(prob)
    print(probas)
    return probas


def main():
    tok = Tokenizer()
    # test_file = 'data/04-input.txt'
    test_file = args.test_file

    # probas = load_model('./data/wiki_ja_train_trained_model')
    probas = load_model(args.lm)
    tok.set_dictionary(probas)

    tokenized_results = []
    with open(test_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            sent = l.rstrip()
            print('Input:', sent)

            print('Tokenized:', tok.tokenize(sent))

            # You can also call the viterbi one by one.
            node_list = tok.forward(sent, probas)
            words = tok.backward(sent, node_list)
            if len(words) == 0:
                words = sent
            print('Tokenized:', words)
            tokenized_results.append(' '.join(words))

    dump_file = args.dump_file
    with open(dump_file, 'w') as f:
        print('save file: {}'.format(dump_file))
        f.writelines(tokenized_results)


if __name__ == '__main__':
    main()
