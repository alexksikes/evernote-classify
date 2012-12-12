How to implement synching?
--------------------------

Here is the idea in implementing this feature. I think this feature would make
this program quite useful.

There would only be one command called "classify" which would always start by
synchronizing the notes from the user online account to a local store. Geeknote
uses SQL like + SQL Alchemy, so we would add a Note model to hold all the
notes. After synching, we can now proceed as before, that is we would build a
dataset, train and evaluate.

The reason why a database is preferred to saving the notes in files is because
we need to keep track of the guid of the note as well as some other metadata.
For example we would run into an issue if two notes in the user account would
have the same title.

We would probably also have a "--clean" command to remove all the notes and
start fresh. Other options are now possible such as:

- Train on classified but evaluate on unclassified.
- Evaluate on n most recently updated notes with training on the rest.
- Evaluate on n random notes with training on the rest.

Offline testing from a directory of files can be performed but in this case we
would just return the cross-validated performance of the classifier. This is
always useful if we want to test the procedure with existing datasets without
having to create an account, login, upload, etc ...

Files of interest:
-----------------

- storage.py (create Note model, same field as Evernote API Note, see if
  possible to put object in db with same fields directly)
- gnsync.py (sync_db, get_notes_db, set_notes_db)

Interface:
----------

    $ geeknote classify 
        [--updated|created|random <how many>] 
        [--notebooks <list of notebook names>] 
        [--auto-assign]
        [--estimate-perf] 
        [--clean]
        
    default: $ geeknote --updated 5

Some more notes
---------------

- sync: get notes notebook by notebook (no need to use fetch_notes)
- eval: fetch notes and build train and test set (exclude gui or notebook
guid from select clause in Note model)

Commit Message
--------------

Added synching feature (no need to re-download)

- added synching from the user online account to a local store
- changed command line with only one command
- added evaluation on created and random notes
- added --estimate-perf command
- added the ability to cleanup the db from the downloaded notes
- changed README for the command