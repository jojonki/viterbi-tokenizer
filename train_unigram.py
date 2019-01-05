from ngram import Ngram


def test():
    ng = Ngram()

    # Your n-gram model is trained with a text file
    ng.train('data/wiki-ja-train.word')

    # You can save your trained model as text. Currently, we do not support loading trained model.
    ng.dump('trained/wiki_ja_train_trained_model', n=1)



if __name__ == '__main__':
    test()
