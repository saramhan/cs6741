import numpy as np
import matplotlib.pyplot as plt
from collections import Counter,defaultdict
import pandas as pd
#%matplotlib inline
#%pylab inline
import gzip
import re, math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
#t=568454
path = 'data' #change the path
def readfile():
    reviews = []
    r = []
    with gzip.open(path,'rb')  as f:
        #F = f.readlines()
        F = [str.decode(f, "UTF-8", "ignore") for f in f.readlines()]
        for n,(l1,l2) in enumerate(zip(F[:], F[1:]+[''])):
            #print l1
            l1 = l1.split('\n')[0]
            l2 = l2.split('\n')[0]
            if l1 == '':
                continue
            if ': ' not in l2:
                l = l1+l2
            elif ': ' not in l1:
                continue
            else:
                l = l1
            r.append(l)
            if len(r) ==8:
                review = {'productId':'', 'userId':'','profileName':'',
                   'helpfulness':'','score':'','time':'','summary':'','text':''}
                for l in r:
                    s = l.split(': ')
                    if s[0] == 'product/productId':
                        review['productId'] = s[1]
                    elif s[0] == 'review/userId':
                        review['userId'] = s[1]
                    elif s[0] == 'review/profileName':
                        review['profileName'] = s[1]
                    elif s[0] == 'review/helpfulness':
                        review['helpfulness'] = s[1]
                    elif s[0] == 'review/score':
                        review['score'] = s[1]
                    elif s[0] == 'review/time':
                        review['time'] = s[1]
                    elif s[0] == 'review/summary':
                        review['summary'] = s[1]
                    elif s[0] == 'review/text':
                        review['text'] = s[1]

                reviews.append(review)
                r = []
    reviews = pd.DataFrame(reviews)
    reviews = reviews.sort_values(by='time')#sort by chronological order
    reviews = reviews.to_dict('records')
    return reviews

reviews = readfile()


def prod_text_fun():
    prod_text = defaultdict()
    for review in reviews:
        prod_text.setdefault(review['productId'], []).append(review['text'])
    return prod_text

def prod_helpful_fun():
    prod_helpful = defaultdict()
    '''Instead of using the helpfulness as a fraction (ex, 2/3),
    I subtracted the non-helpful votes from the helpful votes. (ex, 2- (3-2) = 1)'''
    for review in reviews:
        r = review['helpfulness'].split('/')
        prod_helpful.setdefault(review['productId'], []).append(eval(r[0]) - (eval(r[1])-eval(r[0]))  )
    return prod_helpful

prod_text = prod_text_fun()
prod_helpful = prod_helpful_fun()
Helpfulness_reviewNo_thereshold = 50

def Cos_sim_fun():
    '''return the cosine similarity and reviews
    since the most helpful review was created for each product'''
    tfidf_vectorizer = TfidfVectorizer()
    text_since_mosthelpful = defaultdict()
    simil_since_mosthelpful = defaultdict()
    avg = []
    for i in range(0, 1000000):
        avg.append([0., 0.])
    for k in prod_helpful.keys():
        if len(prod_helpful[k]) > Helpfulness_reviewNo_thereshold: #Get the
          #  print "tag"
            tfidf_matrix_target = tfidf_vectorizer.fit_transform(prod_text[k])
            simMatrix = cosine_similarity(tfidf_matrix_target, tfidf_matrix_target)
         #   print simMatrix.shape
            for idxpair,val in np.ndenumerate(simMatrix):
                if idxpair[0] > idxpair[1]:
#                    print "tag"
                    avg[idxpair[0] - idxpair[1]][0] += val
                    avg[idxpair[0] - idxpair[1]][1] += 1.
            i = prod_helpful[k].index(max(prod_helpful[k]) )
           # tfidf_matrix_target = tfidf_vectorizer.fit_transform(prod_text[k][i:])
            simlist = cosine_similarity(tfidf_matrix_target[i:i + 1], tfidf_matrix_target[i:])
            simlist = simlist.tolist()[0]
            simlist = simlist[1:]
            simil_since_mosthelpful[k] = simlist #
            text_since_mosthelpful[k] = prod_text[k][i+1:]
    return text_since_mosthelpful, simil_since_mosthelpful, avg

text_since_mosthelpful, simil_since_mosthelpful, average_cos_sim= Cos_sim_fun()
print average_cos_sim[1]


def set_array_cre_n(set_n):
    '''return similarity lists as an array'''
    ratings = []
    for pid,v in simil_since_mosthelpful.items():
        if pid in set_n:
            ratings.append(simil_since_mosthelpful[pid])
    max_len = max([len(r) for r in ratings])
    ratings_array = []
    for n,rate in enumerate(ratings):
        ratings_array.append(np.append(np.array(map(float, rate)), np.zeros(max_len- len(rate)) ) )
    return np.array(ratings_array)

#set_1_array = set_array_cre(set_1)
def get_means(r):
    '''return the average of n'th similarity as array'''
    r[np.where(r == 0)] = np.nan
    return np.nanmean(r, axis = 0)

def category_n(ll,ul):
    '''return product ids that have the number of reviews
    after the most helpful review between ll and ul '''
    pids = []
    for pid,v in simil_since_mosthelpful.items():
        if len(v) >= ll and len(v) < ul:
            pids.append(pid)
    return pids

#plot for three different groups
set_3_lower = 20
set_3_upper = 70
set_3 = category_n(set_3_lower, set_3_upper)
set_4_lower = 70
set_4_upper = 130
set_4 = category_n(set_4_lower, set_4_upper)
set_5_lower = 130
set_5_upper = 10000000
set_5 = category_n(set_5_lower, set_5_upper)
set_3_means = get_means(set_array_cre_n(set_3))
set_4_means = get_means(set_array_cre_n(set_4))
set_5_means = get_means(set_array_cre_n(set_5))
avg = []
for val in average_cos_sim:
    if val[1] == 0.:
        val[1] = 1.
    avg.append(val[0]/val[1])
plt.subplot(4,1,1)
plt.plot(range(len(set_3_means)), set_3_means, '-o')
plt.plot(range(len(set_3_means)), avg[1:len(set_3_means) + 1], 'ro')
plt.subplot(4,1,2)
plt.plot(range(len(set_4_means)), set_4_means, '-o')
plt.plot(range(len(set_4_means)), avg[1:len(set_4_means) + 1], 'ro')
plt.subplot(4,1,3)
plt.plot(range(len(set_5_means)), set_5_means, '-o')
plt.plot(range(len(set_5_means)), avg[1:len(set_5_means) + 1], 'ro')

#plot for all
set_all = category_n(0, 10000000)
set_all_means = get_means(set_array_cre_n(set_all))
plt.subplot(4,1,4)
plt.plot(range(len(set_all_means)), set_all_means, '-o')
plt.plot(range(len(set_all_means)), avg[1:len(set_all_means) + 1], 'ro')
plt.show()
