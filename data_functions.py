from datasets import load_dataset
import json
import os
from tqdm import tqdm
import pandas as pd
import datasets
import sys
from transformers import pipeline
import torchtext
from nltk.tokenize import word_tokenize
import nltk
import re
import string
tqdm.pandas()



def concatSents(group):

    '''Concatenates sentences of one video together.'''

    fullTrans = ' '.join(group['text'])
    return fullTrans


def getSummary(x
               ,pipe
               ,first_n_sent = 20
               ):
    
    '''gets the summary of the transcript.

    x: transcript
    pipe: transformers pipeline
    first_n_sent: Integer representing the number of sentences to consider when generating the summary of the transcript.
    
    '''

    subset = getFirstNSents(x,first_n_sent = first_n_sent)

    summ = pipe(subset
                , max_length=50
                , min_length=10
                , do_sample=False
                , truncation=True)[0]['summary_text']
    
    return summ


def getFirstNSents(x
                   ,first_n_sent):
    
    '''Returns the first n sentences of transcript.
    
    x: transcript
    first_n_sent: Integer representing the number of sentences of the transcript to return
    
    '''

    punctuation_indices = []
    for index, char in enumerate(x):
        if char in '!.?':
            punctuation_indices.append(index)
    
    try:
        idx = punctuation_indices[first_n_sent-1]
        return x[0:idx+1]
    except:
        return x


def tokenize(x):

    '''This function
    
        1) adds the start and end of sentence tokens (<sos>, <eoe>) to the end of each sentence in the transcript
        2) splits a string into a list of tokens
        3) converts tokens to lowercase.

    x: transcript

        '''

    if x[-1] not in '!.?':
        x = x + '.'

    punctuation_indices = []
    for index, char in enumerate(x):
        if char in '!.?':
            punctuation_indices.append(index)
    
    if len(punctuation_indices) > 0:
        newTrans = ''
        currIdx = 0
        for idx in punctuation_indices:
            currSent = '<sos> ' + x[currIdx:idx] + ' ' + x[idx] + ' <eos>'
            newTrans = newTrans + ' ' + currSent
            currIdx = idx + 1
        
        result = [t.lower() for t in simple_split(newTrans)]

        return result
    
    else:

        result = [t.lower() for t in simple_split('<sos> ' + x + ' <eos>')]

        return result



def sklearn_split(text):

    '''This function splits a string into a list of tokens.
    
    text: string to split
    '''

    text_trans = re.findall(r'(?u)\b\w\w+\b|[!.?]|[<\w+>]+', text)
    return text_trans


def simple_split(text):

    '''This function splits a string into a list of tokens.
    
    text: string to split
    '''

    text_trans = text.split()
    return text_trans
  

def load_data(
                path='jamescalam/youtube-transcriptions'
                ,split='train'
                ,n_rows=-1
                ,summarize_first_n_sents = 20):

    '''This function loads the dataset and processes it.
    
    path: path to the Hugging Face dataset
    split: only option available for youtube-transcriptions is 'train'
    n_rows: number of rows to return
    summarize_first_n_sents: integer representing how many sentences to consider when generating the summary
    
    '''

    # first download the dataset
    data = load_dataset(
                        path=path,
                        split=split
                    )

    df = pd.DataFrame(data)

    # columns to group by
    groupCols = ['title'
                ,'published'
                ,'url'
                ,'video_id'
                ,'channel_id'
                ]
    
    groupColIdx = [i for i,v in enumerate(groupCols)]
    
    # dictionary that ties a column name to an integer
    groupDic = dict(zip(groupCols,groupColIdx))
    
    # group by columns and concatenate sentences to get the full transcript
    results = []
    for name, group in tqdm(df.groupby( list(groupDic.keys()) )):
        processed_value = concatSents(group)
        results.append({'category': name, 'full_transcript': processed_value})

    # if n_rows = -1, then return all rows
    if n_rows == -1:
        df_agg = pd.DataFrame(results)
    # else return the specified number of rows
    else:
        df_agg = pd.DataFrame(results).head(n_rows)

    # extract the groupby columns from the group name (i.e. category)
    for c in groupDic.keys():
        df_agg[c] = df_agg.category.apply(lambda x: x[groupDic[c]] )

    # drop the category (i.e. name of the group)
    df_agg.drop(columns=['category'], inplace=True)

    # generate the summary of the transcript
    pipe = pipeline("summarization", model="facebook/bart-large-cnn")
    df_agg['summary'] = df_agg['full_transcript'].progress_apply(lambda x: getSummary(x, pipe=pipe, first_n_sent = summarize_first_n_sents))

    # tokenize the transcript
    df_agg['transcript_tokens'] = df_agg['full_transcript'].apply(lambda x: tokenize(x))

    # tokenize the transcript summary
    df_agg['summary_tokens'] = df_agg['summary'].apply(lambda x: tokenize(x))

    return df_agg


def buildVocab(
                iterator
                ,min_freq
                ,specials = ['<unk>', '<pad>', '<sos>', '<eos>']
                ):
    '''
    This function builds a vocab from an iterator.

    iterator: Iterator used to build Vocab. Must yield list or iterator of tokens.
    min_freq: The minimum frequency needed to include a token in the vocabulary.
    specials: Special symbols to add. 

    '''
    vocab = torchtext.vocab.build_vocab_from_iterator(
                                                        iterator=iterator
                                                        ,min_freq=min_freq
                                                        ,specials=specials
                                                        )

    vocab.set_default_index(0)

    return vocab





df = load_data(n_rows=10, summarize_first_n_sents=25)
df.head(10)
df['summary_tokens'][0]
df['summary'][0]

summVocab = buildVocab( df.summary_tokens, min_freq=0)

summVocab.get_itos()[:20]
summVocab(["will", "you", "download", "the", "new", "game", "?"])
summVocab.lookup_tokens(summVocab.lookup_indices(["will", "you", "download", "the", "new", "game", "?"]))



