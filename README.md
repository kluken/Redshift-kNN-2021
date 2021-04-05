# Redshift Estimation using k-Nearest Neighbours or Random Forest (paper In Prep.)

## Requirements:
* Numpy (Ver: 1.14.3)
* Matplotlib (Ver: 2.2.2)
* Astropy (Ver: 2.0.7)
* Sklearn (Ver: 0.19.1)
* TQDM (Ver: 4.23.0)
* PyLMNN (Ver: 1.5.2)
* Metric_Learn (Ver: 0.3.0)

## Usage:
* Using arguments:
  * -a | --algorithm. The type of ML to use. Options: 0 for kNN, 1 for Random Forest 
  * -t | --testType. The training/testing regime to use. Options: 0 for normal, 1 for sub-field test with train = ELAIS-S1, 2 = sub-field test with train = eCDFS
  * -d | --distType. The distance metric to use with the kNN Algorithm. Options: 1 for Manhattan, 2 for Euclidean, 99 for Mahalanobis. 
    * Note. Use the Euclidean Distance Metric when using a learned distance metric.   
  * -b | --bootstrapSize. Number of bootstrap intervals. Do not use if you don't want bootstrapped error bars. 
  * -c | --classification. Whether to use Classification-based algorithms. Any value will trigger the classification mode of the algorithm chosen. 
  * -m | --metricLearn. Should metric learning be used? Don't enter for no. 
  * -k | --kNeighbours. Number of neighbours to use. 
    * Note. Using this option means the value of *k* used by the kNN algorithm or the number of trees used by the Random Forest will not be chosen based on the lowest outlier rate.  
 
 ## Examples:
 * `Redshift.py -a 0 -t 0 -d 2`
  * Using this will run the kNN algorithm in regression mode, using a random training sample and the Euclidean Distance Metric
* `Redshift.py -a 0 -t 1 -d 2 -c True`
  * Using this will run the kNN algorithm in classification mode, using the ELAIS-S1 field as a training set and the Euclidean Distance Metric
* `Redshift.py -a 0 -t 2 -d 99`
  * Using this will run the kNN algorithm in regression mode, using the eCDFS field as a training set, and the Mahalanobis distance metric. 
* `Redshift.py -a 0 -t 0 -d 2 -m True -b 1000`
  * Using this will run the kNN algorithm in regression mode, using a random training sample and the Euclidean Distance metric after transforming the data using the MLKR distance metric. 95% confidence intervals will be computed using 1000 bootstrap samples. 
*   `Redshift.py -a 0 -t 0 -d 2 -m True -c True`
  * Using this will run the kNN algorithm in classification mode, using a random training sample and the Euclidean Distance metric after transforming the data using the LMNN distance metric.    
* `Redshift.py -a 1 -t 0`
  * Using this will run the Random Forest algorithm in regression mode, using a random training sample.  
* `Redshift.py -a 1 -t 0 -c True`
  * Using this will run the Random Forest algorithm in classification mode, using a random training sample.
