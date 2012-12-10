geeknote-classify
=================

Geeknote-classify is a fork of [Geeknote][0] which adds automatic
classification of your Evernote notes into notebooks. At the moment the
program only provides recommendations of last five recently updated notes.

## Installation

Download the repository:

    $ git clone git://github.com/alexksikes/geeknote-classify.git
    $ cd geeknote-classify

Install the excellent [scikit-learn][2] machine learning library:

    $ pip install -U scikit-learn

## Example Usage

First you need to login using geeknote:

    $ python geeknote.py login

Then train a classifier:

    $ python geeknote.py train
    
By default "train" will only fetch the last **300** recently updated notes from
your account. Use "--counts" to override this behavior. You can also save the
fetched notes with the "--save-notes" command (useful for offline testing, see
below).

You can now see the recommendation of the classifier:

    $ python geeknote.py classify

## Testing Everything Works

Two datasets are provided. The first one comes from [Teddy Note][3] and the
second one from a sample of the [20 newsgroup dataset][4].

You can directly test these datasets:

    $ python classifier.py ./classifier_dataset/teddy.train ./classifier_dataset/teddy.test
    
You should get something like that:

    ...
    Making a dataset from the notes saved in classifier_dataset/teddy.train.
    Training ...
    Estimating model performance with 5 folds cross validation ...
    Accuracy: 0.93 (+/- 0.02)
    Making a dataset for the notes saved in classifier_dataset/teddy.test.
    Evaluating the classifier on this dataset.
    Showing the suggestions (current notebook >> suggested notebook).
     0 : 07.12.2012  travel >> travel    An Cabo Verde Windsurf Center
     1 : 07.12.2012  debate >> debate    Should U.S. Continue Aid to Pakistan? - Council on Foreign Relations
     2 : 07.12.2012  debate >> debate    Writing From a Global Perspective
     3 : 07.12.2012  useful >> travel    Getting Started
     4 : 23.03.2012  debate >> debate    The Long War Journal - Charts on US Strikes in Pakistan
    
Additionally you can login into "sandbox_classify" and "sandbox_classify2"
(password is the same as login). "sandbox_classify" is the account holding the
Teddy Note dataset. "sandbox_classify2" holds the 20 newsgroup dataset.

## Future / todo

- not just the 5 update notes ... more options
- more datasets for testing, read from file
- better documentation (doctest)
- use logging instead of print
- use scikit standard namespace
- train on the classified notes, suggest on the unclassified ones
- instead of downloading everything, sample a set of notes
- randomize order of fetch notes for training
- auto assign to notebooks
- integrate with existing unit testing framework
- grid search of best param of classifier
- notes with diff kind of content (not just text)

[0]: http://www.geeknote.me/
[1]: http://www.geeknote.me/install/
[2]: http://scikit-learn.org/
[3]: http://notedev.com/teddy/
[4]: http://qwone.com/~jason/20Newsgroups/
