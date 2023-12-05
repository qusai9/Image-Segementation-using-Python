"""
A library to load the data
"""
import sys
import os
import glob
import numpy as np
import pickle



# def loaddata( dir_name, extension, val_perc, eval_perc ):
#### load a file
def load_file(fname):
  filesize = os.path.getsize(fname)
 # if filesize == 0: print("The file is empty: "
  if filesize == 0:
    print('the file is empty!')
  else:
    with open(fname, 'rb') as fp:
      X_pos, X_neg = pickle.load(fp)
  return X_pos, X_neg

#### Load a whole directory of data...
def load_dir(dir_name):
  fnames = glob.glob(os.path.join(dir_name, "*.pickle"))
  X_pos_all, X_neg_all = load_file(fnames.pop(0))
  for fname in fnames:
    X_pos, X_neg = load_file(fname)
    X_pos_all = np.hstack((X_pos_all, X_pos))
    X_neg_all = np.hstack((X_neg_all, X_neg))
  return X_pos_all.T, X_neg_all.T

# training data
# we will use the training data
# remember we don't care about the negative samples here just the postives
# X_pos_train, _ = load_dir("./DUMP_YCbCr/train")
# we could transpose it here or we can do it in the formula above.
# I did it above.

# validation data
# we care about both the positive and negative samples.
#X_pos_valid, X_neg_valid = load_dir("./DUMP_YCbCr/valid")
# create the labels
# X_pos_valid_labels = np.ones((X_pos_valid.shape[0], 1))
# X_neg_valid_labels = np.zeros((X_neg_valid.shape[0], 1))
  # return the list of files as a tuple
  # return X_pos_train, (X_pos_valid,X_pos_valid_labels), (X_neg_valid,X_neg_valid_labels)
