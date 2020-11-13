import schedule
import time
import datetime
import math
import numpy as np
import requests
import pandas as pd
from bs4 import BeautifulSoup

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from wordcloud import WordCloud,ImageColorGenerator

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score

import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from flask import Flask,request,render_template

def Extraction():
    link1='https://www.amazon.in/Boya-Omnidirectional-Lavalier-Condenser-Microphone/product-reviews/B076B8G5D8/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews'

    total_review=[] 
    total_rating=[]
    total_date=[]

    while 1:
    
        amazon='https://www.amazon.in'
        headers={'User-Agent':'Chrome/84.0.4147.105'}
        response = requests.get(link1,headers=headers)
        soup = BeautifulSoup(response.content,'html.parser')    
            #print(link1)
        review_final = soup.find_all("span",{"data-hook":"review-body"})
        review_rating = soup.find_all("i",{"data-hook":"review-star-rating"})
        review_date=soup.find_all("span",{"data-hook":"review-date"})

    
        for i in range(0,len(review_rating)):
            total_review.append(review_final[i].get_text())
            total_rating.append(review_rating[i].get_text())
            total_date.append(review_date[i].get_text())

    
        x='https://www.amazon.in/Boya-Omnidirectional-Lavalier-Condenser-Microphone/product-reviews/B076B8G5D8?pageNumber=501&reviewerType=all_reviews'
        if link1!=x:
            for next_page in soup.findAll('li', attrs={'class':'a-last'}):
                link_new = next_page.find('a')['href']
            link1= amazon + link_new
            previous=link1
        else:
            break


    for i in range(len(total_review)):
        total_review[i]=total_review[i].lstrip('\n')
        total_review[i]=total_review[i].rstrip('\n')

    for i in range(len(total_rating)):
        total_rating[i]=total_rating[i][0]

    for i in range(len(total_rating)):
        total_date[i]=total_date[i][21:]

    boya=pd.DataFrame()
    boya['REVIEWS']=total_review
    boya['RATINGS']=total_rating
    boya['DATE']=total_date

    return boya

def preproscessing(micro):
    micro['RATINGS']=micro['RATINGS'].replace({'1':'VERY NEGATIVE','2':'NEGATIVE','3':'NEUTRAL','4':'POSITIVE','5':'VERY POSITIVE'})

    
    lemmit= WordNetLemmatizer()

    tokens=[]

    for i in range(len(micro)):
    
        review = re.sub('[^a-zA-Z]', ' ', micro['REVIEWS'][i])
        review=re.sub(' +',' ',review)
        review = review.lower()
        review= re.sub(r'(.)\1+', r'\1\1',review)
        review = review.split()
        review = [lemmit.lemmatize(word) for word in review if not word in stopwords.words('english')]
        review= ' '.join(review)
        tokens.append(review)

    for x in range(len(tokens)):
        micro=micro.replace(to_replace=micro['REVIEWS'][x],value=tokens[x])



    positive_words=[]
    negative_words=[]



    def vade(reviews):
    
        #print(sentiment)
        reviews=reviews.split()
        words=SentimentIntensityAnalyzer()
    
        for i in reviews:
            if(words.polarity_scores(i)['compound'])>=0.01:
                positive_words.append(i)
            
            elif(words.polarity_scores(i)['compound'])<0.01 and (words.polarity_scores(i)['compound'])>-0.01:
                pass
                
            elif(words.polarity_scores(i)['compound'])<=0.01:
                negative_words.append(i)

    micro['REVIEWS'].apply(vade)
    return micro,positive_words,negative_words
    
def Visulaization(micro,positive_words,negative_words):
    
    corpus=[]
    corpus_positive=[]
    corpus_negative=[]


    #Wordcloud for whole corpus
        
    for i in range(len(micro)):
        words=nltk.word_tokenize(micro['REVIEWS'][i])
        corpus.append(words)
    wordcloud=WordCloud(background_color="black").generate(str(corpus))
    plt.figure(figsize=[20,20])
    plt.imshow(wordcloud)
    plt.axis("off")
        #plt.show()
        
#WordCloud for Postive words
    for pos in micro['REVIEWS']:
        pos=pos.split()
        pos_new=[ z for z in pos if z in positive_words]
        corpus_positive.append(pos_new)
    wordcloud_positive=WordCloud(background_color="black").generate(str(corpus_positive))
    plt.figure(figsize=[20,20])
    plt.imshow(wordcloud_positive)
    plt.axis("off")
        #plt.show()



#WordCloud for Negative words
    for neg in micro['REVIEWS']:
        neg=neg.split()
        neg_new=[ z for z in neg if z in negative_words]
        corpus_negative.append(neg_new)
    wordcloud_negative=WordCloud(background_color="black").generate(str(corpus_negative))
    plt.figure(figsize=[20,20])
    plt.imshow(wordcloud_negative)
    plt.axis("off")
        #plt.show()


    sns.countplot(micro['RATINGS'],order=micro['RATINGS'].value_counts().index,palette='dark')
    plt.title('Count Of Each Ratings ')
        #plt.show()


    sns.barplot(x=micro['RATINGS'].value_counts(normalize=True).index,y=micro['RATINGS'].value_counts(normalize=True)*100,palette='colorblind')
    plt.xlabel('Ratings')
    plt.ylabel('Percentage')
    plt.title('Percentage of Ratings')


def Modelling(micro):

        #TF_IDF
    x_train,x_test,y_train,y_test=train_test_split(micro.iloc[:,0],micro.iloc[:,1],test_size=0.01,random_state=0)
    tf_idf=TfidfVectorizer()
    x_train_tdf=tf_idf.fit_transform(x_train)
    x_test_tdf=tf_idf.transform(x_test)

    pickle.dump(tf_idf,open('transformtions.pkl','wb'))
    #Handling Imbalance
    
    from imblearn.over_sampling import RandomOverSampler

    x_random,y_random=RandomOverSampler().fit_resample(x_train_tdf,y_train)
    x_trains_bal,x_test_bal,y_trains_bal,y_test_bal=train_test_split(x_random,y_random,test_size=0.30,random_state=0)
    
    estimators=[("svc",SVC(kernel="linear",random_state=42,gamma=0.08)),('tree',ExtraTreesClassifier(random_state=0)),('multi',LogisticRegression(max_iter=500)),('log',LinearSVC(random_state=42))]
    stack=StackingClassifier(estimators=estimators,final_estimator=SVC(kernel="linear",gamma=0.08)).fit(x_trains_bal,y_trains_bal)


    p_tr_stack=stack.predict(x_trains_bal)  
    p_te_stack=stack.predict(x_test_bal)

    #print(p_tr_stack)
    #print(p_te_stack)

    tr_report_stack=classification_report(y_trains_bal,p_tr_stack)
    te_report_stack=classification_report(y_test_bal,p_te_stack)

    #print('Train ACC')
    #print(tr_report_stack)
    #print('Test ACC')
    #print(te_report_stack)
    pickle.dump(stack,open('nlp_models.pkl','wb'))



    Data=pd.DataFrame()
    change=datetime.datetime.now()
    new_date=change.strftime('%d-%B-%Y')
    current_date=re.sub("[-]"," ",new_date)
    z=[]
    week_data=[]
    for i in range(1,8):
        previous=datetime.datetime.now()-datetime.timedelta(days=i)
        previ_date=previous.strftime('%d-%B-%Y')
        previous_date=re.sub("[-]"," ",previ_date)
        x=micro.loc[micro['DATE']==previous_date,'REVIEWS']
        z=[]
        if len(x)>0:
            z=stack.predict(tf_idf.transform(micro.loc[micro['DATE']==previous_date,'REVIEWS']))
        for y in z:
            week_data.append(y)
        else:
            pass

    Data['WEEK']=week_data

    df=pd.DataFrame({'week':Data['WEEK'].value_counts()},index=['NEGATIVE','NEUTRAL','POSITIVE','VERY NEGATIVE','VERY POSITIVE'])
    plot=df.plot.pie(y='week',figsize=(7,7),title=previous_date+' - '+current_date,autopct='%1.1f%%')
    plt.show()

    fig=plt.figure(figsize=(8,5))
    sns.countplot(x=Data['WEEK'],order=Data['WEEK'].value_counts().index,palette='bright')
    plt.title(previous_date+' - '+current_date)
    plt.show()


    return None


if __name__=='__main__':
    micro=Extraction()
    processed_reviews,pos,neg=preproscessing(micro)
    Visulaization(processed_reviews,pos,neg)
    Modelling(processed_reviews)

