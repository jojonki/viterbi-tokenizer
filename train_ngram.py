import argparse
from ngram import Ngram

parser = argparse.ArgumentParser()
parser.add_argument('--N', type=int, default=1, help='N of n-gram')
parser.add_argument('--train_file', type=str, metavar='PATH', help='train file path')
parser.add_argument('--dump_file', type=str, metavar='PATH', help='tokenized result file')
args = parser.parse_args()

def test():
    ng = Ngram()

    # Your n-gram model is trained with a text file
    # ng.train('data/wiki-ja-train.word')
    ng.train(args.train_file)

    # You can save your trained model as text. Currently, we do not support loading trained model.
    # ng.dump('trained/wiki_ja_train_trained_model', n=1)
    ng.dump('{}-{}gram'.format(args.dump_file, args.N), n=args.N)


if __name__ == '__main__':
    test()
