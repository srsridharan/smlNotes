from numba import autojit

import numpy as np
import cPickle as cpk
import sklearn as skl
import nltk as nltk
from sklearn.feature_extraction.text import CountVectorizer 
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer 

"""
stages of extracting features from a text document:

##word_ngrams(tokenize(preprocess(self.decode(doc))), stop_words)
    - decode: (if chosen) Decode the input into a string of unicode symbols               
    - preprocess : convert to smaller case and strip accents(ascii or unicode or nothing)
    - tokenize
    - spelling correction (optional!!! need to do this if you want)
    - stopwords
    - n-grams : if chosen turn tokens into a sequence of n-grams after stop words filtering. Default n =1

- stemming 

- converting to a vector (transform/fit-transform function)
    - if the mechanism used to do this is to count the number of occurances of the word in a document it is called count vectorizer
    - if the mechanism is tfidf then it is a tfidf vectorizer

NOTE: custom stemmers can be passed into the preprocessing  or tokenization stage of the construction of the 
    countVectorizer or tfidfVectorizer. However if this is done, then the normalization and tokenization functions would have
    to be done by us. A way to circumvent this is to derive a new class (say stemCountVectorizer class or a stemtfidfVectorizer class) from the basic
    vectorizer class

The basic count vectorizer does NOT normalize the feature vectors BUT the TFIDF does normalize the feature vectors
"""


eng_stemmer = SnowballStemmer('english') 

#@autojit
class StemmedCountVectorizer(CountVectorizer):
    def build_analzyer(self):
        analyzer = super(StemmedCountVectorizer,self).build_analyzer()
        return lambda doc:(eng_stemmer.stem(w) for w in analyzer(doc))



class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer() 
        return lambda doc: (eng_stemmer.stem(w) for w in analyzer(doc)) 


def usetfidf():
    vectorizer = StemmedTfidfVectorizer(min_df=1,stop_words='english', decode_error='ignore')
    return vectorizer

def usecountVec():
    vectorizer = StemmedCountVectorizer(min_df=1,stop_words='english', decode_error='ignore')
    return vectorizer

def findClosest(X,y):    
    proxComp = []
    for num,x in enumerate(X):
        proxComp.append([np.linalg.norm((x-y).toarray()),num])
    proxComp = sorted(proxComp)    
    return proxComp

def studyOfNormalization():
    #===========================================================================
    # This function demonstrates that 
    #  1. the standard countVectorizer (true for both with and witout stemmer) does not normalize
    # the feature vector (i.e. repetitions of the entire sentence will change the result). 
    #  2. the tfidf vectorizer (true for both with and without stemmer) DOES normalize the feature vector.
    #  
    #===========================================================================
    
    #setOfDocs=["this is one two three","this is three four five this is three four five","this is five six seven"]
    setOfDocs=["this is oneA twoA threeA","this is threeA fourA fiveA","this is threeA fourA fiveA this is threeA fourA fiveA","this is fiveA sixA sevenA"]
    newDoc=['this is twoa threea sevena'] 
    
    # now find the doc closest to the new doc in setOfDocs
    
    vectorizerTFIDF = usetfidf()
    vectorizerCountvec = usecountVec()
    listOfVectorizers= [vectorizerTFIDF,vectorizerCountvec]
    listOfnameOfVectorizer = ['vectorizerTFIDF','vectorizerCountvec']
    
    for vectorizer,nameOfVectorizer in zip(listOfVectorizers,listOfnameOfVectorizer):
        X = vectorizer.fit_transform(setOfDocs)
        y = vectorizer.transform(newDoc)
        # find closest doc
        orderOfProximity = findClosest(X,y)
        print '\nUsing vectorizer: ', nameOfVectorizer
        print vectorizer.get_feature_names()
        print 'closeness to ',newDoc
        for prox, k in orderOfProximity:
            print prox,": ",setOfDocs[k]
        #=======================================================================
        # import ipdb
        # ipdb.set_trace()
        #=======================================================================


def loadNewsGroupDataAndSaveToPkl():
    #===========================================================================
    # This function loads the newsgroup data sets from files and converts them to pkl files.
    # note: this is done hoping that future loads of the pkl files would be faster than loading from the text files on
    # disk
    #===========================================================================
    
    import sklearn.datasets 
    MLCOMP_DIR = r"/Volumes/disk0s3/Datasets/"
    #===========================================================================
    # data = sklearn.datasets.load_mlcomp("20news-18828",mlcomp_root=MLCOMP_DIR)
    #===========================================================================
    #===========================================================================
    # train_data = sklearn.datasets.load_mlcomp("20news-18828", "train", mlcomp_root=MLCOMP_DIR)
    # print(train_data.filenames) 
    #===========================================================================
    groups = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space'] 
    print "loading newsgroup training data...."
    train_data = sklearn.datasets.load_mlcomp("20news-18828", "train", mlcomp_root=MLCOMP_DIR, categories=groups)    
    
    print 'saving to pkl file ...'
    # write the train_data to a trainDataFile
    fh = open("./trainDatapkl.pkl","wb")
    cpk.dump(train_data,fh)
    fh.close()

    # read the train data and write  it into a pkl file
    print 'loading test data ...'
    test_data = sklearn.datasets.load_mlcomp("20news-18828", "test", mlcomp_root=MLCOMP_DIR,categories = groups)
    print 'saving to pkl file ...'
    fh = open("./testDatapkl.pkl","wb")
    cpk.dump(test_data,fh)
    fh.close()
            
def newsGroupPklDataLoader(): 
    #===========================================================================
    # Loads the train and test data for the newsgroup data set from pkl files
    #===========================================================================
    

    fh = open("./trainDatapkl.pkl","rb")
    train_data =  cpk.load(fh)
    fh.close()
    
    # read the train data
    fh = open("./testDatapkl.pkl","rb")
    test_data =  cpk.load(fh)
    fh.close()
    return [train_data,test_data]

def clusterData(vectorizedTrainData,numClusters):
    # returns trained km
    
    from sklearn.cluster import KMeans
    print 'running the km clustering...'
    km = KMeans(n_clusters=numClusters, init='random', n_init=1, verbose=0) 
    km.fit(vectorizedTrainData)
    return km

def saveTrainedKMForNewsGroups(trainedKM):
    fh = open('trainedKMNgroup.pkl',"wb")
    cpk.dump(trainedKM,fh)
    fh.close()

def loadTrainedKMForNewsGroups(): 
    fh = open('trainedKMNgroup.pkl',"rb")
    trainedKM = cpk.load(fh)
    fh.close()
    return trainedKM
 
def createTrainedVectorizer(traindataset,vectorizer):
    # returns :[txData,vectorizer]
    txData = vectorizer.fit_transform(traindataset.data)
    return [vectorizer,txData]
def saveNewsGroupVectorizer(vectorizer):
    fh = open('trainednewgroupvec.pkl',"wb")
    cpk.dump(vectorizer, fh)
    fh.close()
def loadNewsGroupVectorizer():
    fh = open('trainednewgroupvec.pkl',"rb")
    vectorizer =  cpk.load(fh)
    fh.close()
    return [vectorizer[0],vectorizer[1]]

if __name__=="__main__":
    #studyOfNormalization()
    # newsGroupsClustering()
    #import pdb as pdb
    #pdb.set_trace()
   
    #===========================================================================
    # to load the entire newsgroup data set (rather than as train/test)
    
    # data = sklearn.datasets.load_mlcomp("20news-18828",mlcomp_root=MLCOMP_DIR)
    #===========================================================================

    
    #loadNewsGroupDataAndSaveToPkl()
    trainData,testData = newsGroupPklDataLoader()
    vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5,stop_words='english', decode_error='ignore') 
    [vectorizer,txData] = createTrainedVectorizer(trainData,vectorizer)
    saveNewsGroupVectorizer([vectorizer,txData])
    
    trainedVectorizer,txVecData = loadNewsGroupVectorizer()
    num_clusters = 50
    trainedKM = clusterData(txVecData,num_clusters)
    saveTrainedKMForNewsGroups(trainedKM)
    #===========================================================================
    trainedKM = loadTrainedKMForNewsGroups()
    new_post = testData.data[20]
    print 'post to find similar posts to', new_post
    new_post_vec = trainedVectorizer.transform([new_post]) 
    new_post_label = trainedKM.predict(new_post_vec)[0]
    similar_indices = (trainedKM.labels_==new_post_label).nonzero()[0]
    similar = [] 
    for i in similar_indices:
        dist = np.linalg.norm((new_post_vec - txVecData[i]).toarray())
        similar.append((dist, trainData.data[i])) 
        similar = sorted(similar) 
    
    print 'most similar: ', similar[0]
    print 'least similar: ',similar[-1]
    
