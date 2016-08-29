# # Create your views here.


from django.shortcuts import render_to_response
from django.shortcuts import render
from .forms import PostForm
from django.template import RequestContext
from django.http import JsonResponse
import csv
import re
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB


def home(request):
    return render(request, 'home.html')

def work(request):
    return render(request,'work.html')

TRAIN_SET="/home/nimble33/mysite/static/trainingData/trainActual.csv"
def post_new(request):
    #classCount,wordClassCount=divideTraining(TRAIN_SET)
    if request.method == 'POST': # If the form has been submitted...
        form = PostForm(request.POST) # A form bound to the POST data
        if form.is_valid(): # All validation rules pass
            # Process the data in form.cleaned_data
            # ...

            #print(form.cleaned_data['text'])
            test_review=form.cleaned_data['text']
            '''
            Classifier Goes Here
            '''
            review,rating=divideTraining(TRAIN_SET)
            #result=predictSvm(test_review,review,rating)
            result=predictNb(test_review,review,rating)

            #print("first item")
            # #print(review[0])
            # result=predictNB(classCount,wordClassCount,test_review)
            # print(result)
            # #return render('home.html', {'result': result})
            # #data=result
            return JsonResponse({'result':result[0]})


            #return HttpResponseRedirect('/thanks/') # Redirect after POST
    else:
        form = PostForm() # An unbound form

    return render_to_response('tefy.html', {'form': form, }, context_instance = RequestContext(request))

trainReview=[]
trainPosNegRating=[]
def divideTraining(trainSet):
	with open(trainSet, 'rU') as f:
		lines = csv.reader(f)
		for line in lines:
			if len(line) == 2:
				trainReview.append(line[1])
				trainPosNegRating.append(line[0])
		#trainingSize=len(trainReview)
		return trainReview,trainPosNegRating

def predictSvm(testreview,trainReviews,trainRatings):
    classifierSVM = svm.LinearSVC()
    textClassifierSVM = Pipeline([('vect', CountVectorizer(stop_words='english')),
								 ('tfidf', TfidfTransformer()),
								 ('clf', classifierSVM), ])
    textClassifierSVM = textClassifierSVM.fit(trainReviews, trainRatings)
    predictedResults=textClassifierSVM.predict(testreview)
    return predictedResults

def predictNb(testreview,trainReviews,trainRatings):
    classifierNb = MultinomialNB()
    textClassifierNb = Pipeline([('vect', CountVectorizer(stop_words='english')),
								 ('tfidf', TfidfTransformer()),
								 ('clf', classifierNb), ])
    textClassifierNb = textClassifierNb.fit(trainReviews, trainRatings)
    predictedResults=textClassifierNb.predict(testreview)
    return predictedResults

# def divideTraining(trainSet):
#     # print 'Naive Bayes training'
#     with open(trainSet, 'rU') as f:
#         lines = csv.reader(f)
#         for line in lines:
#             if len(line) == 2:
#                 #Positive/Negative or [1-5]
#                 classLabel = line[0]
#                 classCountDictionary=classCount(classLabel)
#                 #Split by white spaces
#                 wordSplit = re.split('\s+', line[1].lower())
#                 wordNoStopWords = removeStopWords(wordSplit)
#                 for word in wordNoStopWords:
#                     wordClassLabelCountDictionary=wordClassCount(word,classLabel)
#                     #print wordClassLabelCountDictionary
#     return classCountDictionary,wordClassLabelCountDictionary

# finalProbability={}
# def predictNB(targetLabelCount,wordLabelCount,testReview):
#     testResults=[]
#     #Total Size
#     totalReviewsInDataSet=sum(targetLabelCount.values())

#     #print totalReviewsInDataSet
#     #Labels either [positive,negative] or [1,2,3,4,5]
#     classLabels = targetLabelCount.keys()
#     #Explore test set
#     wordSplit = re.split('\s+', testReview.lower())
#     # print "Tokenized:"
#     # print wordSplit
#     wordNoStopWords = removeStopWords(wordSplit)
#     # print "With no stop words"
#     # print wordNoStopWords
#     #Arrange as list
#     words=list(wordNoStopWords)
#     # print "words in line"
#     # print words
#     #Mapping each word to a classlabel
#     for classLabel in classLabels:
#         '''
#         #1. Calculation of Prior: from the training set
#         p(class)=no. of reviews classified to that class/total number of reviews
#         p(positive)=positively classified count/total count
#         p(negative)=negatively classified count/total count
#         '''
#         # Positive/Negative Scenario
#         countReviewSpecificClassLabel=getClassCount(classLabel,targetLabelCount)
#         priorProbability=getPriorProbability(countReviewSpecificClassLabel,totalReviewsInDataSet)
#         # print "prior probability:"
#         # print classLabel
#         # print priorProbability
#         '''
#         2. Calculation of Likelihood


#         i. Individual word probabilities = word occurence/classcount
#         ii. Likelihood= product of individual word probabilities

#         '''
#         wordProbability = []
#         #Calculating Individual word probabilities p(word/class)
#         for word in words:
#             wordOccurence=getWordOccurence(word,classLabel,wordLabelCount)
#             classCount=getClassCount(classLabel,targetLabelCount)
#             wordProbability.append(wordOccurence/classCount)
#         # print "word probability"
#         # print wordProbability


#         #Product of individual word probabilities
#         likelihood=1.0
#         for indivWordProb in wordProbability:
#             likelihood*=indivWordProb

#         # print "likelihood"
#         # print classLabel


#         '''
#         3. Calculation of Final Probability
#         '''
#         finalProbability[classLabel] = likelihood * priorProbability
#     positiveScore = finalProbability.get('positive')
#     negativeScore = finalProbability.get('negative')
#     if (positiveScore > negativeScore):
#         testResults.append('positive')
#     else:
#         testResults.append('negative')
#     return testResults


# classesCountDictionary={}
# wordClassCountDictionary={}

# #Increase count for every label
# #Stores class:count Eg ->{'positive': 15787, 'negative': 4213}
# def classCount(classLabel):
#     #classesCountDictionary[classLabel]=classesCountDictionary.get(classLabel,0)+1
#     if classLabel not in classesCountDictionary:
#         classesCountDictionary[classLabel]=0
#     classesCountDictionary[classLabel]+=1
#     return classesCountDictionary

# # 'boiled': {'positive': 24, 'negative': 7}
# def wordClassCount(word,classLabel):
#     if not word in wordClassCountDictionary:
#         wordClassCountDictionary[word] = {}
#     if classLabel not in wordClassCountDictionary[word]:
#         wordClassCountDictionary[word][classLabel]=0
#     wordClassCountDictionary[word][classLabel]+=1
#     return wordClassCountDictionary

# def removeStopWords(wordList):
#     #Stopwords: Words that have no value in the context of sentiment analysis
#     stopWordList = stopwords.words('english')
#     stopWordList=[item.encode('utf-8') for item in stopWordList]
#     # stopWordList.remove('not')
#     # stopWordList.remove('no')
#     # stopWordList.remove('against')
#     newList=[]
#     for word in wordList:
#         if word not in stopWordList:
#             newList.append(word)
#     return newList

# #Returns the prior probability which is
# def getPriorProbability(part,total):
#     return part/total


# #Returns class count of given class label Example <- getClassCount('positive;) - 15787
# def getClassCount(classLabel,targetLabelCount):
#     return targetLabelCount.get(classLabel)

# def getWordOccurence(word,classLabel,wordLabelCount):
#     if word in wordLabelCount.keys():
#         wordInTrainSet = wordLabelCount[word]
#         if classLabel in wordInTrainSet.keys():
#             return wordInTrainSet[classLabel]
#         else:
#             return 0.000000001
#     else:
#         return 0.000000001





