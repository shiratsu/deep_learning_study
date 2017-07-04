# coding: UTF-8

from itertools import chain
import pycrfsuite
import sklearn
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

from corpusreader import CorpusReader

import get_more_info

def bio_classification_report(y_true, y_pred):
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )


if __name__ == "__main__":

    c = CorpusReader('hironsan.txt')
    test_sents = c.iob_sents('test')

    X_test = [get_more_info.sent2features(s) for s in test_sents]
    y_test = [get_more_info.sent2labels(s) for s in test_sents]

    tagger = pycrfsuite.Tagger()
    tagger.open('model.crfsuite')

    print(test_sents[0])

    print("-----------------------------------------")

    # テスト（まずは一個例を）
    example_sent = test_sents[0]
    print(' '.join(get_more_info.sent2tokens(example_sent)))

    print("Predicted:", ' '.join(tagger.tag(get_more_info.sent2features(example_sent))))
    print("Correct:  ", ' '.join(get_more_info.sent2labels(example_sent)))


    y_pred = [tagger.tag(xseq) for xseq in X_test]

    print(bio_classification_report(y_test, y_pred))
