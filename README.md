### COMP90049 Project1 Lexical Normalization Based on Tweets
This is the Project 1 for COMP90049 (Knowledge Technologies) from the University of Melbourne.

This project implements three machine learning models, including Naïve Bayes, Multinomial Logistic Regression and Random Forest, based on WEKA for tweets sentiment analysis. For more details, please check the [project specifications](https://github.com/Andy-TK/COMP90049_Project2/blob/master/2019S1-90049P2-spec.pdf) and [project report](https://github.com/Andy-TK/COMP90049_Project2/blob/master/COMP90049%20Project%202%20Report.pdf).


#### What is WEKA?
Weka is a collection of machine learning algorithms for data mining tasks. It contains tools for data preparation, classification, regression, clustering, association rules mining, and visualization. Weka also supports deep learning.

<a href="https://www.cs.waikato.ac.nz/~ml/weka/index.html" target="_blank"><img src="https://github.com/Andy-TK/COMP90049_Project2/blob/master/Models/WEKA_LOGO.jpg" alt="WEKA Logo" width="200"></a>

Weka is open source software issued under the [GNU General Public License](http://www.gnu.org/licenses/gpl-3.0.html).

Here are several free online courses that teach machine learning and data mining using Weka. The videos for the courses are available on [Youtube](https://www.youtube.com/user/WekaMOOC). For more details, please check [Weka Manual](https://github.com/Andy-TK/COMP90049_Project2/blob/master/WekaManual.pdf).

Download WEKA for free: https://www.cs.waikato.ac.nz/~ml/weka/downloading.html

#### 1. Data
* `traineval.arff`
> A combination of given training set and evaluation set, which is used for training models in WEKA based on 10-folds cross validation. There are 47 attributes including tweets id, some linguistic vocabularies and sentiment label.

* `test.arff`
> is used to test the performance of models. The sentiment attribute in this file is set as ‘?’.

* `test-tweets.txt`, `traineval-labels.txt` and `traineval-tweets.txt` 
> are used in `topic_analysis.py` for the sentiment analysis of users based on tweets text and given topics.

#### 2. Code
* `roc_plot.py`
> is used to plot the ROC curves of three machine learning models based on the relevant ROC result csv files (e.g. `negative_NB.csv`) exported from WEKA.

* `test_labels.py`
> is used to extract test labels from the result csv file (e.g. `test_MLR.csv`) and write it into a txt file (e.g. `test-labels-MLR.txt`) with tweet ids for the further topic analysis and save it into folder "Moels".

* `topic_analysis.py`
> is used to is used for tweets sentiment analysis based on tweets text and a given topic.

#### 3. Models
The WEKA Knowledge Flow Layout is shown as below:
![WEKA Knowledge Flow](https://github.com/Andy-TK/COMP90049_Project2/blob/master/Models/WEKA_Knowledge_Flow.jpg "WEKA Knowledge Flow")

* `WEKA_KF.kf`
> the WEKA Knowledge Flow layout file.

* `model_MLR.txt`
> contains the model information of Multinomial Logistic Regression.

* `model_NB.txt`
> contains the model information of Naïve Bayes.

* `model_RF.txt`
> contains the model information of Random Forest.

* `test_MLR.arff`
> the test result with arff format of Multinomial Logistic Regression.

* `test_MLR.csv`
> the test result with csv format of Multinomial Logistic Regression.

* `test_NB.arff`
> the test result with arff format of Naïve Bayes.

* `test_NB.csv`
> the test result with csv format of Naïve Bayes.

* `test_RF.arff`
> the test result with arff format of Random Forest.

* `test_RF.csv`
> the test result with csv format of Random Forest.

* `test-labels-MLR.txt`
> the test-labels result of Multinomial LR used in `topic_analysis.py`.

* `test-labels-NB.txt`
> the test-labels result of Naïve Bayes used in `topic_analysis.py`.

* `test-labels-RF.txt`
> the test-labels result of Random Forest used in `topic_analysis.py`.

#### 4. ROC
The negative ROC results of three models plotted through `roc_plot.py`:
![Negative ROC](https://github.com/Andy-TK/COMP90049_Project2/blob/master/ROC/ROC_negative.png "Negative ROC")

The neutral ROC results of three models plotted through `roc_plot.py`:
![Neutral ROC](https://github.com/Andy-TK/COMP90049_Project2/blob/master/ROC/ROC_neutral.png "Neutral ROC")

The positive ROC results of three models plotted through `roc_plot.py`:
![Positive ROC](https://github.com/Andy-TK/COMP90049_Project2/blob/master/ROC/ROC_positive.png "Positive ROC")

* `negative_MLR.csv`
> WEKA negative ROC results with csv format of Multinomial LR.

* `negative_NB.csv`
> WEKA negative ROC results with csv format of Naïve Bayes.

* `negative_RF.csv`
> WEKA negative ROC results with csv format of Random Forest.

* `neutral_MLR.csv`
> WEKA neutral ROC results with csv format of Multinomial LR.

* `neutral_NB.csv`
> WEKA neutral ROC results with csv format of Naïve Bayes.

* `neutral_RF.csv`
> WEKA neutral ROC results with csv format of Random Forest.

* `positive_MLR.csv`
> WEKA positive ROC results with csv format of Multinomial LR.

* `positive_NB.csv`
> WEKA positive ROC results with csv format of Naïve Bayes.

* `positive_RF.csv`
> WEKA positive ROC results with csv format of Random Forest.

* `negative_MLR.arff`
> WEKA negative ROC results with arff format of Multinomial LR.

* `negative_NB.arff`
> WEKA negative ROC results with arff format of Naïve Bayes.

* `negative_RF.arff`
> WEKA negative ROC results with arff format of Random Forest.

* `neutral_MLR.arff`
> WEKA neutral ROC results with arff format of Multinomial LR.

* `neutral_NB.arff`
> WEKA neutral ROC results with arff format of Naïve Bayes.

* `neutral_RF.arff`
> WEKA neutral ROC results with arff format of Random Forest.

* `positive_MLR.arff`
> WEKA positive ROC results with arff format of Multinomial LR.

* `positive_NB.arff`
> WEKA positive ROC results with arff format of Naïve Bayes.

* `positive_RF.arff`
> WEKA positive ROC results with arff format of Random Forest.


