#! /usr/bin/env python
import cPickle as pickle
import sys
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn import cross_validation
from sklearn import datasets

from geeknote import out
from lib.html2text import html2text
from tools import to_unicode

# features are bag of words weighted by tf-idf
# for high dimensional and sparse (text data), we use a linear classifier
def get_model():
    return Pipeline([
         ('vect', CountVectorizer()),
         ('tfidf', TfidfTransformer()),
         ('clf', LinearSVC()),
    ])
    
# there seems to be a limit in the number of notes returned to 50!
# so we page every 50 steps updating the offset
def list_notes(gn, limit, start_offset, step=50):
    def do(offset):
        lim = step if limit > step else limit #! we may get a bit more result
        result = gn.findNotes('updated:20000101', lim, offset=offset, updateOrder=True)
        offset += step
        return result, offset
    
    result, offset = do(start_offset)
    limit = result.totalNotes if limit > result.totalNotes else limit
    while offset < start_offset + limit:
        r, offset = do(offset)
        result.notes.extend(r.notes)
        
    return result    
        
def fetch_notes(gn, limit, offset=0, save_notes=None, cache=True):
    # get the list of most recently updated notes
    result = list_notes(gn, limit, offset)
    o = gn.getStorage()
    o.setSearch(result)
    
    # get the notebook name assigned to each note
    notebooks = dict((nbk.guid, nbk) for nbk in gn.findNotebooks())
    
    # get the notes ENML content and add notebook name
    for i, note in enumerate(result.notes):
        note.notebookName = notebooks[note.notebookGuid].name
        
        # avoid re-downloading note content
        content = o.getNoteContent(note.guid) if cache else None
        
        if content and note.updated == content.updated:
            note.content = content.content
        else:
            gn.loadNoteContent(note)
            o.setNoteContent(note) if cache else None
            
        show_note_snippet(note, i)
    
    # save the fetched notes for offline testing
    if save_notes:
        print 'Saving fetched notes to %s.' % save_notes
        pickle.dump(result, open(save_notes, 'wb'))
    return result

def train_evernote(gn, count, save_notes=None, clear_cache=False):
#    out.preloader.stop()
    out.preloader.setMessage('')
    if clear_cache:
        print 'Clearing the note content cache.'
        gn.getStorage().clearNotesContent()
    print 'Fetch recently updated notes, omitting first 5, limiting to %s notes.' % count
    result = fetch_notes(gn, count, 5, save_notes)
    print 'Making a dataset from all these notes.'
    data = make_dataset(result)
    print 'Training on the examples.'
    model = train(data.examples, data.targets)
    print 'Saving the model. Now you can use "classify".'
    save_classifier(gn, model, data.target_names)
    return model
    
def classify_evernote(gn, save_notes=None):
#    out.preloader.stop()
    out.preloader.setMessage('')
    print 'Fetch the 5 recently updated notes.'
    result = fetch_notes(gn, 5, 0, save_notes)
    print 'Making a dataset from these 5 notes.'
    data = make_dataset(result)
    print 'Evaluating the classifier on these notes.'
    model = get_classifier(gn)
    if not model:
        print 'You first need to train a classifer. Use "train" command.'
        return
    targets = evaluate(model, data.examples)
    print 'Showing the suggestions (current notebook >> suggested notebook).'
    show_notes_suggestion(result, targets, model.target_names)

def save_classifier(gn, model, target_names):
    model.target_names = target_names
    gn.getStorage().setUserprop('classifier', model)

def get_classifier(gn):
    return gn.getStorage().getUserprop('classifier')
    
def show_note_snippet(note, key=0):
    notebook = note.notebookName if hasattr(note, 'notebookName') else ''
    suggested = (' >> ' + note.suggestedNotebook) if hasattr(note, 'suggestedNotebook') else ''
    out.printLine("%s : %s%s%s" % (
        str(key).rjust(2, " "),
        #print date
        out.printDate(note.updated).ljust(12, " ") if hasattr(note, 'updated') else '',
        #print notebook
        (notebook + suggested + ' ').ljust(20, " ") if (notebook or suggested) else '',
        #print title
        note.title if hasattr(note, 'title') else note.name,
    ))

def show_notes_snippet(result):
    for i, note in enumerate(result.notes):
        show_note_snippet(note, i)
    
def show_notes_suggestion(result, targets, target_names):
    for i, (note, target) in enumerate(zip(result.notes, targets)):
        note.suggestedNotebook = target_names[target]
        show_note_snippet(note, i)

# we get the title, tags and stripped HTML
# content coming from Evernote is expected to be UTF-8 encoded
def note_to_text(note):
    content = html2text(to_unicode(note.content))
    title = note.title.decode('utf-8')
    if note.tagNames:
        tags = ('\n'+' | '.join(note.tagNames)).decode('utf-8')
    else:
        tags = ''
    return title + tags + '\n' + content
    
def make_dataset(result, categories=[]):
    # the notebook names are the different targets / classes
    #! sklearn may provide something here
    target_names = []
    for n in result.notes:
        label = n.notebookName
        if label not in target_names:
            target_names.append(label)
    
    # text of the notes as features, notebook name as target
    examples, targets = [], []
    for n in result.notes:
        text = note_to_text(n)
        examples.append(text)
        label = n.notebookName
        targets.append(target_names.index(label))
        
    # return the dataset used for the classifier
    #! use same convention as sklearn datasets 
    return datasets.base.Bunch(
        examples=examples, 
        targets=np.array(targets, dtype=int),
        target_names=target_names
    )

def train(examples, targets, perf=True):
    model = get_model()
    # show cross validated score of the model
    if perf:
        show_model_perf(model, examples, targets)
    # train the model on the all data
    model.fit(examples, targets)
    return model

def show_model_perf(model, examples, targets, cv=5):
    print 'Estimating model performance with %s folds cross validation ...' % cv
    try:
        scores = cross_validation.cross_val_score(model, examples, targets, cv=cv)
        print 'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() / 2)
    except ValueError as e:
        print '>> Warning: Could not estimate cross validated model performance.'
        print '>> Reason: %s' % e

def evaluate(model, examples):
    #! how to tell the classifier to return ints for targets?
    targets = model.predict(examples)
    return np.array(targets, dtype=int)

# you must first download a train and test set using --save-notes option
# or use the dataset provided in ./classifier_dataset
def test(train_path, test_path):
    train_notes = pickle.load(open(train_path, 'rb'))
    print 'Showing notes on which we train.'
    show_notes_snippet(train_notes)
    print 'Making a dataset from the notes saved in %s.' % train_path
    data_train = make_dataset(train_notes)
    print 'Training ...'
    model = train(data_train.examples, data_train.targets)
    
    test_notes = pickle.load(open(test_path, 'rb'))
    print 'Making a dataset for the notes saved in %s.' % test_path
    data_test = make_dataset(test_notes)
    print 'Evaluating the classifier on this dataset.'
    targets = evaluate(model, data_test.examples)
    print 'Showing the suggestions (current notebook >> suggested notebook).'
    show_notes_suggestion(test_notes, targets, data_train.target_names)
    
if __name__ == '__main__':
    test(sys.argv[1], sys.argv[2])
