# NLP-Toxic-Comment
[Kaggle Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)

## Goal
- Create a toxic comment classifier

## Data
- [Kaggle](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data)
- `all_data.csv`

## Steps
- Import data to Pandas DataFrame
- Explore and analyze unbalanced data
  - (Optional) Run PCA analysis 
- Preprocess data
  - Preprocess toxic data to binary: `toxicity, severe_toxicity, obscene, sexual_explicit, identity_attack, insult, threat`
  - Preprocess comment text data
  - (Optional) Display WordCloud
  - Create balanced dataset
- Build Logistic Regression model
  - Split data to train and test
  - Add stopwords
  - Add tokenizing, TF-IDF
- Compare F1 scores of these ML models
  - Logistic Regression
  - KNN
  - SVM
  - Bayes
  - Random Forest
- Predict sample toxic data
- GridSeach for best regularization parameter
- Plot ROC curve and compute AUC score



