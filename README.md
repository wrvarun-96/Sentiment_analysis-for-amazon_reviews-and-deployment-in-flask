# Sentiment_analysis-for-amazon_reviews-and-deployment-using-flask

## Libraries that has to be installed.
  1)Beautiful soap
  2)Scikit-learn
  3)Pickle
  4)Flask

## Overview.
  1) Reviews for a specific product are extracted . 
  2) Generated  insights from the text data by applying various text mining methods. 
  3) Sentiments (positive, more positive, negative, more negative and neutral), are determined, by the model deployed, based on the new reviews from the website.

## ML algo used is Stacking Classifier. 
   We created the model using Stacking Classifier where base estimators (SVC,ExtraTreeClassifier,Logistic,LinearSVC) and final estimator(SVC).

## Following are steps how I did the deployment using "Flask".

  1)Have used FLASK for deployment and have deployed Stack Classifier for deployment.
  2)TFIDF and Model as been loaded in disk using pickle.
  3)We will be using @app.route(‘/’) to execute home function which directly goes to home page which is created by html and is named by file home.html.
  4)Then when predict is clicked it goes to @app.route(‘/predict’,methods=[POST]) to execute predict sentiment and given by file final.html.
  5)Getting reviews automatically for past 7 days and getting sentiment of those reviews can be done by entering any product URL which goes to @app.route('/',methods=['GET','POST’]) and scrapes reviews and predict. Followed by getting percentage of each sentiment which can be seen using barplot.
  6)The link for this whole deployment is given by http://127.0.0:5000. This only works in local machine.
