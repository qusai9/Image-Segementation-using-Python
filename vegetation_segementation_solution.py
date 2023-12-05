"""
In this the main script for part one of the final assignment
we will classify data in texture_full using two way of classification
"""
import sys
import numpy as np
# our libraries
from libs import dataloader

if __name__ == "__main__":
  # the arguments to pass when running the code
  # in our case only the image that we want to be segmented
  # args = sys.argv

  # load the data ( location, flag )
 # T, V, E = dataloader.loaddata(args[1], args[2])
  X_pos_train, X_neg_train = dataloader.load_dir("./PART_I_RGB_DATA/train")
  X_pos_valid, X_neg_valid = dataloader.load_dir("./PART_I_RGB_DATA/valid")
  X_pos_eval, X_neg_eval = dataloader.load_dir("./PART_I_RGB_DATA/eval")
  # inspect the data that has been loaded.
  # for i in range(5):
  #   print(X_neg_train[i, :])
  print( 'Train data', np.shape(X_pos_train) , np.shape(X_neg_train) )
  print( 'Valid data', np.shape(X_pos_valid), np.shape(X_neg_valid))
  print('Eval data', np.shape(X_pos_eval), np.shape(X_pos_eval))
  # create the labels
  X_pos_valid_labels = np.ones((X_pos_valid.shape[0], 1))
  X_neg_valid_labels = np.zeros((X_neg_valid.shape[0], 1))

  # # now let's create the feature vector
  # if len( args ) < 6:
  #   print( 'not enough arguments to extract - exiting' )
  #   sys.exit( 1 )
  # ori = int( args[5])
  # cpb = (int( args[6]), int(args[6]))
  # ppc = (int( args[7]), int( args[7]))
  # bbow = conv_bool( args[8])
  # clusters = int( args[9])
  # # extract the hog features
  # # hog_features(to_extract, orientations=8, cells_p_block=(1, 1), pixel_p_cell=(16, 16)):
  # T, V, E = features.extract_hog(T, V, E,
  #                                orientations=ori,
  #                                cells_p_block=cpb,
  #                                pixel_p_cell= ppc,
  #                                as_bow=bbow,
  #                                clusters=clusters)
  # # now let's classify the data
  # if len( args ) < 11:
  #   print( 'not enough arguments to classify and evaluate - exiting' )
  #   sys.exit( 1 )
  # # create the svm
  # # training, validation, C, gamma
  # C = [10**c for c in range(int(args[10]), int(args[11]) + 1 )]
  # gamma = [10**g for g in range(int(args[12]), int(args[13]) + 1 )]
  # svm = classify_evaluate.rbf_train_valid(T, V, C, gamma)
  # # now we will classify the data using the evaluation set.
  # classify_evaluate.evaluate(svm, E)