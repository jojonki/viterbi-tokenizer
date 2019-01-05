# viterbi-tokenizer

## How to Use

### Train n-gram
At first, you need to train n-bram based language model.
```
$ python train_ngram.py --train_file data/wiki-ja-train.word --N 2 --dump_file trained/wiki_ja_train_trained_model
```

By using the trained language model, you can test the tokenizer.
```
$ python tokenize_test.py --lm ./trained/wiki_ja_train_trained_model-1gram --test_file data/wiki-ja-test.txt --dump_file my_answer_wiki-ja.word
```

Finally, you can evaluate your tokenization with Graham's script.
```
$ ./script/gradews.pl data/wiki-ja-test.word my_answer_wiki-ja.word
Sent Accuracy: 0.00% (/84)
Word Prec: 48.94% (1576/3220)
Word Rec: 68.31% (1576/2307)
F-meas: 57.03%
Bound Accuracy: 71.64% (2311/3226)
```

## Reference
- [NLP tutorial 04 by Graham Neubig](http://www.phontron.com/slides/nlp-programming-ja-03-ws.pdf)
