import pylab


def gbm_learning_curve(clf):
    pylab.plot(range(len(clf.oob_score_)), clf.oob_score_, 'o-r', range(len(clf.oob_score_)), clf.train_score_, 'o-b')
