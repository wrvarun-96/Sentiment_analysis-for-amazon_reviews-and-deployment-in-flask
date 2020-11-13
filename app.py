from flask import Flask,request,render_template
from sklearn.ensemble import StackingClassifier
import numpy as np
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import io
import base64
from bs4 import BeautifulSoup
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

import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64 
import os

app=Flask(__name__)

stack_bal=pickle.load(open('nlp_models.pkl','rb'))
trans=pickle.load(open('transformtions.pkl','rb'))

total_review=[] 
total_rating=[]
total_date=[]

picfolder=os.path.join('static','images')
app.config['UPLOAD_FOLDER']=picfolder


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method=='POST':
        message=request.form['message']
        data=[message]
        vect=trans.transform(data).toarray()
        model_predict=stack_bal.predict(vect)
        print(model_predict)

    return render_template('final.html',prediction=model_predict)

@app.route('/',methods=['GET','POST'])
def week():
	if request.method == 'POST':
	  	link1=request.form['prod']
	  	for i in range(1,501):
	  		link2="&pageNumber="+str(i)
	  		link3=link1+link2
	  		head={"User-Agent":"Chrome/84.0.4147.105"}
	  		response = requests.get(link3,headers=head)
	  		soup = BeautifulSoup(response.content,'html.parser')
	  		review_final = soup.find_all("span",{"data-hook":"review-body"})
	  		review_rating = soup.find_all("i",{"data-hook":"review-star-rating"})
	  		review_date=soup.find_all("span",{"data-hook":"review-date"})
	  		for i in range(0,len(review_rating)):
	  			total_review.append(review_final[i].get_text())
	  			total_rating.append(review_rating[i].get_text())
	  			total_date.append(review_date[i].get_text())
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
	  	boya['RATINGS']=boya['RATINGS'].replace({'1':'VERY NEGATIVE','2':'NEGATIVE','3':'NEUTRAL','4':'POSITIVE','5':'VERY POSITIVE'})
	  	lemmit= WordNetLemmatizer()
	  	tokens=[]
	  	for i in range(len(boya)):
	  		review = re.sub('[^a-zA-Z]', ' ', boya['REVIEWS'][i])
	  		review=re.sub(' +',' ',review)
	  		review = review.lower()
	  		review= re.sub(r'(.)\1+', r'\1\1',review)
	  		review = review.split()
	  		review = [lemmit.lemmatize(word) for word in review if not word in stopwords.words('english')]
	  		review= ' '.join(review)
	  		tokens.append(review)
	  	for x in range(len(tokens)):
	  		micro=boya.replace(to_replace=boya['REVIEWS'][x],value=tokens[x])
	  	Data=pd.DataFrame()
	  	change=datetime.datetime.now()
	  	new_date=change.strftime('%d-%B-%Y')
	  	current_date=re.sub("[-]"," ",new_date)
	  	week_data=[]
	  	week_review=[]
	  	for i in range(1,8):
	  		previous=datetime.datetime.now()-datetime.timedelta(days=i)
	  		previ_date=previous.strftime('%d-%B-%Y')
	  		previous_date=re.sub("[-]"," ",previ_date)
	  		if previous_date[0]==str(0):
	  			previous_date=previous_date[1:]
	  		else:
	  			pass
	  		x=boya.loc[boya['DATE']==previous_date,'REVIEWS']
	  		z=[]
	  		if len(x)>0:
	  			z=stack_bal.predict(trans.transform(boya.loc[micro['DATE']==previous_date,'REVIEWS']))
	  			for y in z:
	  				week_data.append(y)
	  			for a in x:
	  				week_review.append(a)
	  		else:
	  			pass
	  	Data['REVIEW']=week_review
	  	Data['PREDICT']=week_data
	  	fig=plt.figure(figsize=(12,6))
	  	sns.barplot(x=Data['PREDICT'].value_counts(normalize=True).index,y=Data['PREDICT'].value_counts(normalize=True)*100,palette='bright')
	  	buf = BytesIO()
	  	plt.savefig(buf, format='png')
	  	buf.seek(0)  # rewind to beginning of file
	  	figdata_png = base64.b64encode(buf.getvalue())
	return render_template('week.html',tables=[Data.to_html(classes='data')],titles=Data.columns.values,weeks=figdata_png.decode('utf8'))
        

if __name__=='__main__':
	app.run(debug=True)