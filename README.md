# SMS-Classifier
SMS spam classifier is a machine learning based model that will accurately classify which texts are spam.

This is a machine learning project that classifies SMS messages as either spam or not spam (ham). It uses a dataset of 5,572 SMS messages that have been labeled as spam or ham.

Installation To run this project, you will need Python 3 and the following libraries:

* numpy
* pandas
* scikit-learn
* nltk
* matplotlib
You can install these libraries using pip. For example:

pip install numpy pandas scikit-learn nltk

# **Usage**

Clone the repository to your local machine:

# **Results**

The SMS spam classifier achieved an accuracy of 98.5% on the test set. This means that it correctly classified 98.65% of the SMS messages as either spam or ham.

# **Credit**

The dataset used in this project was obtained from the UCI Machine Learning Repository.

# Neutral Network SMS Text Classifier

![image](https://github.com/Macwhat/SMS-Classifier/assets/116700271/446b92fd-a642-400a-9f3c-e82b84778255)

In this challenge, we will create a machine learning model that will classify SMS messages as either "ham" or "spam". A "ham" message is a normal message sent by a friend. A "spam" message is an advertisement or a message sent by a company..

We need to create a function called predict_message that takes a message string as an argument and returns a list. The first element in the list should be a number between zero and one that indicates the likeliness of "ham" (0) or "spam" (1). The second element in the list should be the word "ham" or "spam", depending on which is most likely.

For this challenge, you will use the SMS Spam Collection dataset. The dataset has already been grouped into train data and test data.

![image](https://github.com/Macwhat/SMS-Classifier/assets/116700271/4ffb90e0-10d0-4b02-96f4-cdeb251f5afd)

As we can see that the dataset contains three unnamed columns with null values. So we drop those columns and rename the columns v1 and v2 to label and Text, respectively. Since the target variable is in string form, we will encode it numerically using pandas function **.map()**.

# HAM and SPAM data

![image](https://github.com/Macwhat/SMS-Classifier/assets/116700271/124a137f-00a6-463d-a4de-33f94d405924)

# **Model Summary**

![image](https://github.com/Macwhat/SMS-Classifier/assets/116700271/187a18c0-6f25-4c2e-9c52-d20c1333d2e2)

# Contributors

### * Mihika Dey

# Problem

We have an unbalanced dataset; most of our data points contain the label “ham,” which is natural because most SMS are ham. Accuracy cannot be an appropriate metric in certain situations. Other measurements are required.


