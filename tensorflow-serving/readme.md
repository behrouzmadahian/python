# Deep Neural Net readme

## 00_chop_data.py

- For demonstration purposes, chops the data to show how to read from multiple files in parallel in training

## 01_int64nparray_to_tfrecords.py

* Convert data into Tensorflow's binary file formats
* Enables to read directly from disk without the need to load the whole data in memory
* enables to read from multiple binary files in training
* Solution to massive data set training- data set size bounded by hard drive space!
* For demonstration purposes, I chopped the train, validation, and test data into 10 chunks each
* Can be ignored if not desired

## 02_hyper_param_optimization.py

* Searches the space of desired hyper parameters.
* Defines a parse function to parse examples read from binary .tfRecords files
* Defines a batching function that batches the examples read from files
* Defines a 3-layer fully connected Neural network
* Defines run_training in which model is trained
  * Puts  inputs signatures on CPU
  * Puts training the model on GPU

* Perform a random search on hyper parameter space
* Saves a checkpoint of search process to be used to continue search in case of interruptions

## 03_train.py

* Uses results of hyper parameter search to train the model
* Puts data signatures on CPU
* Puts training signatures on GPU
* Tracks model performance during training
* Checkpoints trained weights and smoothed version of model weights- exponential moving averages
* Saves the model with best validation performance

## 04_model_load_predict.py

* Restores the check pointed model weights into model
* Loads data from csv files
* Has the option of performing prediction using trained weights or their associated smoothed version
* Performs prediction and saves the results to file



## 05_plot_AUC.py

* Plots the AUC of the false positive rate- true positive rate curve for train, validation, and holdout data sets



## Data:

[link to Rui's data files](\\RUI-LENOVO\Share\Relevance\Data)








