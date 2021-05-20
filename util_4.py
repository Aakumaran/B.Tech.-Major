import os
import os.path
import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import brown as allwords
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential, load_model, model_from_config
import keras.backend as K
import language_tool_python
import spacy
tool = language_tool_python.LanguageTool('en-US')
sp=spacy.load('en_core_web_sm')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
dataset_avg=round(6.799198643858838)


def essay_to_wordlist(essay_v, remove_stopwords):
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)


def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    num_words = 0.
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            num_words += 1
            featureVec = np.add(featureVec, model[word])
    featureVec = np.divide(featureVec, num_words)
    return featureVec


def getAvgFeatureVecs(essay, model, num_features):
    counter = 0
    essayFeatureVecs = np.zeros((1, num_features), dtype="float32")
    essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
    return essayFeatureVecs

def get_model():
    b_size = 32
    i_size = 350
    model = Sequential()
    model.add(LSTM(i_size, dropout=0.5, recurrent_dropout=0.4, input_shape=[1, i_size], return_sequences=True))
    model.add(LSTM(120, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, 120], return_sequences=True))
    model.add(LSTM(b_size, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    return model

def lstm_mark(test_essay):
    b_size = 32
    i_size = 350
    model = KeyedVectors.load_word2vec_format(f"word2vecmodel.bin", binary=True) 
    wlist = essay_to_wordlist(test_essay, remove_stopwords=True)
    testDataVecs = getAvgFeatureVecs(wlist, model,i_size)  # , num_features)
    testDataVecs = np.array(testDataVecs)
    testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

    lstm_model = get_model()
    lstm_model.load_weights(f"lstm_model.h5") 
    y_pred = lstm_model.predict(testDataVecs)
    y_pred = np.around(y_pred)
    m_struct = (y_pred[0][0])
    return m_struct

def gen_fb(mark):
    #1- Very poor 2- Satisfactory 3-Good 4- Very Good 5 - Excellent
    if mark<0.2:
        return "Very Poor"
    elif mark<0.4:   #6.4
        return "Poor"
    elif mark<0.6:
        return "Satisfactory"
    elif mark<0.7:
        return "Good"
    elif mark<0.9:
        return "Very Good"
    else:
        return "Excellent"
        
def gen_senti_fb(mark):
    if mark<-0.6:
        return "Strongly Negative"
    elif mark<-0.3:
        return "Quite Negative"
    elif mark>0.6:
        return "Strongly Positive"
    elif mark>0.3:
        return "Quite Positive"
    else:
        return "Neutral"
        
def getLength(test_essay):
    l=0
    word_list = allwords.words()  #  List of almost all words in English
    word_set = set(word_list)
    stop_words = set(stopwords.words('english'))  #  Set of all English stop words
    word_tokens = word_tokenize(test_essay)
    for word in word_tokens:
         if word not in stop_words:
            if word in word_set:
                l+=1
    return l

def senti_analyse(test_essay):
    sid=SentimentIntensityAnalyzer()
    scores=sid.polarity_scores(test_essay)
    return scores['compound']

def entity_detection(text):
    text_given=sp(text)
    ent_list=[]
    ent_type_list=[]
    for entity in text_given.ents:
        ent_list.append([entity.text, entity.label_])
        ent_type_list.append(entity.label_)
    entity_dict={}
    ent_type_list=list(set(ent_type_list))
    # Creating a list for each entity type present
    for entity_type in ent_type_list:
        entity_dict['{0}'.format(entity_type)]=[]
    # Adding the entities to the corresponding entity_type lists
    for ent in ent_list:
        entity_dict[ent[1]].append(ent[0])
    # Splitting Person's names into words
    try:   #6.4
        if(bool(entity_dict['PERSON'])):   
            for name in entity_dict['PERSON']:
                name_split = name.split()
                entity_dict['PERSON'] = entity_dict['PERSON'] + name_split
                entity_dict['PERSON'].remove(name)
    except:
        pass
    for key in entity_dict:
        entity_dict[key]=list(set(entity_dict[key]))
    return entity_dict

def wiki_factual(test_essay, ref_file, fact_marks):#(test_essay, topic, fact_marks):
    #Essay entities (Total number of facts detected)
    essay_dict=entity_detection(test_essay)
    ref_content=open(ref_file, 'r')
    #save wiki text with topic name
    wiki_text=ref_content.read()
    #Wiki Entities
    wiki_dict=entity_detection(wiki_text)
    # Number of correct facts
    correct_factcount = 0
    for entity_type in essay_dict:
        if entity_type in wiki_dict:
            for entity in essay_dict[entity_type]:
                if entity in wiki_dict[entity_type]:
                    correct_factcount += 1
                else:
                    continue
        else:
            continue
    # Expected facts count
    prop_const = 0
    score_deducted = 0   #6.4
    expected_count = 0   #6.4
    fact_marks=int(fact_marks)
    if fact_marks<6:
        expected_count = int(fact_marks*1)
        prop_const = 1
    elif fact_marks <20:
        expected_count = fact_marks * 0.5
        prop_const = 0.5
    else:
        expected_count = fact_marks*0.2
        prop_const = 0.2
    # Detected facts count
    detected_count = 0
    for key in essay_dict:
        temp = len(essay_dict[key])
        detected_count += temp
    if detected_count<=0:   #6.4
        total_score=0
        return total_score
    #Score calculation
    if detected_count > expected_count:
        if correct_factcount >= expected_cont:
            score_awarded=expected_count/prop_const
            # Penalising for writing extra facts which are wrong
            score_deducted=((detected_count-correct_factcount)/detected_count)/prop_const
        else:
            score_awarded=correct_factcount/prop_const
            # Penalising for writing wrong facts
            score_deducted=((detected_count-correct_factcount)/detected_count)/prop_const
    elif detected_count < expected_count:
        score_awarded=correct_factcount/prop_const
        # Penalising for missing facts
        score_deducted_missing=((expected_count-detected_count)/expected_count)/prop_const
        # Penalising for writing wrong facts
        score_deducted_wrong=((detected_count-correct_factcount)/detected_count)/prop_const
        score_deducted=score_deducted_missing + score_deducted_wrong
    else:
        score_awarded=correct_factcount/prop_const
        score_deducted=((expected_count-correct_factcount)/expected_count)/prop_const
    total_score=score_awarded-score_deducted
    return total_score

def file_factual(test_essay, ref_file, fact_marks):
    #Essay entities (Total number of facts detected)
    essay_dict=entity_detection(test_essay)
    # Reference content
    ref_content=open(ref_file, 'r')
    ref_text=ref_content.read()
    # Reference entities
    ref_dict=entity_detection(ref_text)
    # Number of correct facts
    correct_factcount = 0
    for entity_type in essay_dict:
        if entity_type in ref_dict:
            for entity in essay_dict[entity_type]:
                if entity in ref_dict[entity_type]:
                    correct_factcount += 1
                else:
                    continue
        else:
            continue
    #Expected facts count
    score_deducted = 0
    expected_count = 0
    for keye in ref_dict:
        temp0 = len(ref_dict[keye])
        expected_count += temp0
    #Detected facts count
    detected_count = 0
    for key in essay_dict:
        temp = len(essay_dict[key])
        detected_count += temp
    if detected_count<=0:  #6.4
        total_score=0
        return total_score
    #Score Calculation
    if detected_count > expected_count:
        score_awarded=correct_factcount/expected_count
    elif detected_count < expected_count:
        score_awarded=correct_factcount/expected_count
        #Penalising for missing facts
        score_deducted=(expected_count-detected_count)/(4*expected_count)
    else:
        score_awarded=correct_factcount/expected_count
        
    total_score=score_awarded-score_deducted
    return total_score*fact_marks

def final(ui, ref_type,test_file,ref_file="null"):
    #ui=["0. w_len","1. w_struct","2. w_gram","3. w_fact","4. word_count","5. out_of"]
    is_exceeded=False
    w_total = int(ui[0])+int(ui[1])+int(ui[2])+int(ui[3])
    if w_total<=0:  #6.4  set equal weights
        w_len = 0.25
        w_struct = 0.25
        w_gram = 0.25
        w_fact = 0.25
        w_total=1
    else:        
        w_len = int(ui[0])/w_total
        w_struct = int(ui[1])/w_total
        w_gram = int(ui[2])/w_total
        w_fact = int(ui[3])/w_total
    word_count = int(ui[4])
    if word_count<=0:
        word_count=100 #6.4 default
    
    out_of = int(ui[5])
    if out_of<=0:
        out_of=100  #6.4 default
    
    file1 = open(test_file,"r") 
    test_essay =file1.read()          
    file1.close() 
    
    testw = test_essay.split(" ")
    testl=len(testw)
    matches = tool.check(test_essay)
    
    m_len = (getLength(test_essay)/word_count)        
    if m_len>1:
        is_exceeded=True
        if m_len>1.3:
            m_len=0
        else: 
            m_len = 1.9-m_len
    else:
        is_exceeded=False
    b_struct=lstm_mark(test_essay) #6.4
    if b_struct>3.5:
        m_struct = 1/(1+np.exp(-(b_struct-(dataset_avg/2))))  
    else:
        m_struct = 1/(1+np.exp(-(b_struct-(dataset_avg/2))))  
        
    m_gram= (1-len(matches)/len(testw))  
    m_senti = senti_analyse(test_essay)    
    
    if ref_type==2:  #ref_type=2;from stored/new file
        m_fact = file_factual(test_essay,ref_file,out_of)/out_of
    elif ref_type==1:  #ref_type=1;from wiki
        m_fact = wiki_factual(test_essay,ref_file,out_of)/out_of
    else:   # ref_type=0; no factual analysis
        m_fact=0
    if m_fact<0:  #6.4
        m_fact=0


    tot_mark = out_of*(m_len*w_len+m_struct*w_struct+m_gram*w_gram+m_fact*w_fact)
    if tot_mark>=out_of:  #6.4
        tot_mark=out_of   #6.4
    
    fb=[]
    if(is_exceeded==True):
        fb.append((100*m_len,"Exceeded Word Limit"))
    else:
        fb.append((100*m_len,gen_fb(m_len)))
    fb.append((100*m_struct,gen_fb(m_struct)))
    fb.append((100*m_gram,gen_fb(m_gram)))
    if ref_type==0:
        fb.append((0,"Not opted"))
    else:
        fb.append((100*m_fact,gen_fb(m_fact)))    
    fb.append((m_senti,gen_senti_fb(m_senti)))
    
    return tot_mark, fb

   # script_location = os.path.dirname(os.path.abspath('__file__'))
def marks(ui, flag , file, ref):
    ref_ref_file = ref
    ui_list = ui
    ref_type = flag
    filename = file
    script_location = os.path.dirname(os.path.abspath("lstm_model.h5"))
    test_file = f"{script_location}\\files\\{filename}"
    if flag !=0:
        ref_file = f"{script_location}\\test_files\\{ref_ref_file}"
    else:
        ref_file = "null"
    a = final(ui_list, ref_type,test_file, ref_file)
    return a
