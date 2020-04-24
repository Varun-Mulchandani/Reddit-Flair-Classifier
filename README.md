# Tools

Tensorflow (v1.15.0)

Keras (v2.2.5)

Nltk (stopwords, wordnet)

Seaborn

Matplolib

Wordcloud

Praw

Flask

HTML/CSS

Pickle


# Data Acquisition:
Notebook - reddit_data_acquisition.ipynb

Praw was used to access data from the subreddit r/india.
The following were the attributes utilised:

1)Flair

2)Title

3)Author

While building the application, these attributes we extracted using the url provided.
script - final.py

# Exploratory Data Analysis:
Notebook - EDA.ipynb
Matplotlib, Seaborn and Wordcloud were used.
Takeaway:

1)Most authors, with many submissions, made them with similar flairs.

2)Submissions with flair 'politics' had the highest scores.

3)Initially, each flair had around 200 - 250 submissions making for a well balanced dataset.

4)Submissions with different flairs had similar character level distributions.

5)Submissions with each flair had some common words which was visualised using word cloud. For example:
- Submissions with flair 'Politics' had words 'Government','India','BJP', etc. occuring more often than others.

# Building Models (Training and Testing):
The following are the models I have trained on the data. I decided to stick to the one which gave the highest accuracy while describing how I overcame the various issues faced.

1)Multi channel CNN

2)BERT - english, uncased(24, 1024, 16)

3)Stacked LSTM Network:
Notebook - Final_LSTM.ipynb

Initial Issue: Overfitting (Training accuracy 95% but the validation accuracy would reach 40-45% and then decrease)
Solution:
a) Used synonym based data augmentation to increase training data (Utilised wordnet(nltk))

b) Introduced regularisers.(l1 and l2)

c) Introduced and increased Dropout Rate.(Rate = 0.5)

Validation Split = 0.1

Final accuracy acheived:
Training = 96%

Validation and Test = 84-86%

