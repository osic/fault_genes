from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
import csv
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer, WordNetLemmatizer
import matplotlib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

matplotlib.use('Agg')
url = []
bug_description = []
answer = []
d = {}


def csvParse(file):
    with open(file, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        reader.fieldname = "answer", "summary", "title", "url"
        for row in reader:
            content = row['title']+ " " + row['summary'] + \
                      row['answer']
            d[row['url']] = content.replace("/", " ").replace("-", " ").replace(".", " ")
        return d


def feature_selection(d):
    for key, value in d.iteritems():
        print "Feature Selection bug description corresponding to URL: " + key
        # value = remove_punctuation(value)
        value = tokenize(value)
        value = stemming(value)
        tagged_value = tagger(value)
        reduced_list = wordReduction(tagged_value)
        reduced_string = listToString(reduced_list)
        d[key] = reduced_string
    tfidf_result = tfidfCalc(d)
    dist = calcCosine(tfidf_result)
    clusters = clusterKmeans(tfidf_result,dist)
    frame = constructDataFrame(clusters, d)
    return frame

def constructDataFrame(clusters, d):
    csv_dict = {}
    url_list = []
    description_list = []
    csv_dict['clusters'] = clusters
    for key,value in d.iteritems():
        url_list.append(key)
        description_list.append(value)
    csv_dict['url'] = url_list
    csv_dict['description'] = description_list
    frame  = pd.DataFrame(csv_dict, index = [clusters], columns = ['url','description'])
    return frame

def clusterKmeans(tfidf_matrix, dist):
    from sklearn.cluster import KMeans
    num_clusters = 5
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_matrix)
    clusters  = km.labels_.tolist()
    return clusters

def calcCosine(tfidf_matrix):
    from sklearn.metrics.pairwise import cosine_similarity
    dist = 1 - cosine_similarity(tfidf_matrix)
    return dist

def listToString(fetched_list):
    generated_string = " ".join(str(x) for x in fetched_list)
    return generated_string


def tfidfCalc(d):
    tfidf_dict = {}
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfs = tfidf.fit_transform(d.values())
    #feature_names = tfidf.get_feature_names()
    #for col in tfs.nonzero()[1]:
    #    tfidf_dict[feature_names[col]] = tfs[0, col]
    return tfs


def wordReduction(wordList):
    wordfreq = []
    for word in wordList:
        wordfreq.append(wordList.count(word))
    wordDict = dict(zip(wordList, wordfreq))
    reducedList = []
    reducedList = [key for key, value in wordDict.iteritems() if wordDict[key] > 0]
    return reducedList


def tagger(text):
    text_tagged = nltk.pos_tag(text)
    text_tagged = [t for t in text_tagged if t[1] == "NN" or t[1] == "VBZ"]
    try:
        text_tagged = [str(t[0]) for t in text_tagged]
    except:
        pass
    return text_tagged


def remove_punctuation(text):
    text = text.lower().translate(None, string.punctuation)
    return text


def stemming(text):
    lemmatiser = WordNetLemmatizer()
    count = 0
    for word in text:
        lem_word = lemmatiser.lemmatize(word)
        text[count] = lem_word
        count = count + 1
    return list(set(text))


def tokenize(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    tokens = [w for w in tokens if not w in stopwords.words('english') and w not in "p" and w not in "/p"]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return list(set(filtered_tokens))


def buildCsv(final_dict, out_path):
    #out_file = open(out_path, 'wb')
    #writer = csv.writer(out_file, dialect='excel')
    #for key, value in final_dict.iteritems():
        #writer.writerow([key, value])
    data_frame = pd.DataFrame(final_dict)
    data_frame.to_csv("tagged_ask_os.csv",sep=",")


def main():
    d = csvParse("buglist1.csv")
    d = feature_selection(d)
    buildCsv(d, "tagged_buglist1.csv")


if __name__ == "__main__": main()
