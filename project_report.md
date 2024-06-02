# Project Report: Naive Bayes Classifier for Email Filtering

## FULL DISCLOSURE
ALL CODE IN THIS PROJECT AND THE ENTIRE WRITEUP BELOW WAS GENEREATED BY DEVIN, AN AI SOFTWARE ENGINEER.
I DID NOT WRITE ANY OF THIS CODE, OR THE WRITEUP, BUT RATHER DID THE 4 DAYS WORTH OF PROMPT ENGINEERING TO GUIDE DEVIN.

## Introduction
The objective of this project is to develop a Naive Bayes classifier to accurately distinguish between spam and non-spam (ham) emails. The project involves preprocessing email datasets, implementing the Naive Bayes algorithm, training the model with a given dataset, and evaluating its performance on a separate test dataset. Additionally, the project addresses any class imbalance issues that may affect the model's performance.

## Methodology
### Dataset
The dataset used for this project is the SpamAssassin Public Corpus, which consists of labeled emails divided into spam and ham categories. The dataset was split into training (80%) and testing (20%) sets.

### Preprocessing
The preprocessing steps included:
- Converting all emails to lowercase to ensure case insensitivity.
- Removing all HTML tags and attributes using BeautifulSoup.
- Tokenizing the text into words using NLTK.
- Removing stop words (common words that are unlikely to be useful for classification) using NLTK.
- Performing stemming to reduce words to their base or root form using the PorterStemmer from NLTK.

### Feature Extraction
The feature extraction steps included:
- Converting the processed text into a vector of word counts using the Bag of Words model with CountVectorizer.
- Applying TF-IDF (Term Frequency-Inverse Document Frequency) transformation using TfidfTransformer to reflect the importance of words in emails.

### Naive Bayes Classifier
The Naive Bayes algorithm implemented is the Multinomial Naive Bayes variant, which is suitable for text classification. The classifier was trained using the training dataset, and Laplace smoothing (alpha=1.0) was applied to handle zero probabilities.

### Addressing Class Imbalance
The class distribution revealed a significant imbalance, with 2801 ham emails and only 501 spam emails. To address this issue, the minority class (spam emails) was upsampled to balance the dataset.

## Results
The performance of the Naive Bayes classifier was evaluated using the test dataset. The evaluation metrics calculated were:
- **Accuracy**: The proportion of total emails correctly classified.
- **Precision**: The proportion of predicted spam emails that are actually spam.
- **Recall**: The proportion of actual spam emails that were correctly identified.
- **F1 Score**: The harmonic mean of precision and recall, providing a balance between the two.

The results are as follows:
- Accuracy: 99.29%
- Precision: 98.71%
- Recall: 99.81%
- F1 Score: 99.26%

These metrics indicate that the model is performing exceptionally well in distinguishing between spam and non-spam emails.

## Conclusion
In this project, a Naive Bayes classifier was developed to accurately distinguish between spam and non-spam emails. The preprocessing steps, feature extraction, and implementation of the Multinomial Naive Bayes algorithm were detailed. The class imbalance issue was addressed by upsampling the minority class, resulting in a well-performing model with high accuracy, precision, recall, and F1 score.

Future work could involve exploring other machine learning algorithms, such as Support Vector Machines or deep learning models, to further improve the classifier's performance. Additionally, incorporating more advanced text preprocessing techniques and feature extraction methods could enhance the model's ability to accurately classify emails.

## References
- SpamAssassin Public Corpus: http://spamassassin.apache.org/publiccorpus/
- NLTK: https://www.nltk.org/
- scikit-learn: https://scikit-learn.org/
- BeautifulSoup: https://www.crummy.com/software/BeautifulSoup/
