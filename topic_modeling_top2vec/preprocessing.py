import re, os
import pandas as pd

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

from bs4 import BeautifulSoup
import spacy
import unidecode
from word2number import w2n
import contractions
import logging


logger = logging.getLogger('preprocessing')
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
if not logger.handlers:
    logger.addHandler(sh)

supporting_files_path = './supporting_files/'

custom_stopwords = pd.read_csv(supporting_files_path+'custom_stopwords_reddit.csv', header=None)[0].tolist()


nlp = spacy.load('en_core_web_md')
called = 0
total = 0

# exclude words from spacy stopwords list
deselect_stop_words = []
for w in custom_stopwords:
    nlp.vocab[w].is_stop = True
    
def remove_url_content(text):
    return re.sub(r"http\S+", "", text)


def remove_email_content(text):
    return re.sub('\S*@\S*\s?', '', text)
  
    
def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text


def remove_whitespace(text):
    """remove extra whitespaces from text"""
    text = text.strip()
    return " ".join(text.split())


def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text


def expand_contractions(text):
    """expand shortened words, e.g. don't to do not"""
    text = contractions.fix(text)
    return text


def remove_html_tags(text):
    return re.sub('<[^<]+?>', '', text)


def text_preprocessing(text, accented_chars=True, contractions=True, 
                       convert_num=False, extra_whitespace=True, 
                       lemmatization=True, lowercase=True, punctuations=True,
                       remove_html=True, remove_num=True, special_chars=True, 
                       stop_words=True, remove_url=True, remove_email=True, remove_html_content=True):
    """preprocess text with default option set to true for all steps"""
    global called
    called+=1
    print('Processed ' + str(called) + ' of ' + str(total) + ' records', end='\r', flush=True)

    repl_strings = ['note: this description was reformatted from the original email body. please view the case feed to see the original email body.',
                    'this description was reformatted from the original email body. please view the case feed to see the original email body',
                    '*no content*',
                    '&nbsp;&nbsp;sent from mail for windows 10&nbsp;',
                    'sent from my iphone',
                    '&nbsp;sent from mail for windows 10&nbsp;',
                    'sent from mail for windows 10',
                    '&nbsp;&nbsp;&nbsp;',
                    '&nbsp;&nbsp;sent from mail for windows 10&nbsp;',
                    'window',
                    '&nbsp;',
                    '&nbsp']

    text = text.lower()
    for string in repl_strings:
        text = text.replace(string, '')

    
    if extra_whitespace == True: #remove extra whitespaces
        text = remove_whitespace(text)
    if accented_chars == True: #remove accented characters
        text = remove_accented_chars(text)
    if contractions == True: #expand contractions
        text = expand_contractions(text)
    # if lowercase == True: #convert all characters to lowercase
    #     text = text.lower()
    if remove_url == True:
        text = remove_url_content(text)
    if remove_email == True:
        text = remove_email_content(text)
    if remove_html_content == True:
        text = remove_html_tags(text)
    if remove_html == True: #remove html tags
        text = strip_html_tags(text)

    doc = nlp(text) #tokenise text
    
    # Checking names and removing it
    fil = [i for i in doc.ents if i.label_.lower() in ["person"]]
    for chunks in fil:
        text = text.replace(str(chunks), '')
        
    doc = nlp(text) #tokenise text

    clean_text = []
    
    for token in doc:
        flag = True
        edit = token.text
        # remove stop words
        if stop_words == True and token.is_stop and token.pos_ != 'NUM': 
            flag = False
        # remove punctuations
        if punctuations == True and token.pos_ == 'PUNCT' and flag == True: 
            flag = False
        # remove special characters
        if special_chars == True and token.pos_ == 'SYM' and flag == True: 
            flag = False
        # remove numbers
        if remove_num == True and (token.pos_ == 'NUM' or token.text.isnumeric()) \
        and flag == True:
            flag = False
        # convert number words to numeric numbers
        if convert_num == True and token.pos_ == 'NUM' and flag == True:
            edit = w2n.word_to_num(token.text)
        # convert tokens to base form
        elif lemmatization == True and token.lemma_ != "-PRON-" and flag == True:
            edit = token.lemma_
        # append tokens edited and not removed to list 
        if edit != "" and flag == True:
            clean_text.append(edit)

    return clean_text


def replace(text):
    return text.replace(' ', '_')


# Replacing N_grams
def replace_ngrams(text):
    ngrams = pd.read_csv(supporting_files_path+'ngrams.csv')
    ngrams['mapping_new'] = ngrams['Mapping'].apply(replace)
    
    for i, ngram in enumerate(ngrams['Word'].tolist()):
        if ngram in text:
            text = text.replace(ngram, ngrams.loc[i, 'mapping_new'])
    return text


def join_wrds(text):
    abbr = pd.read_csv(supporting_files_path+'abbr.csv')

    # Abbreviation List
    abbr_list = abbr['Word'].tolist()

    # Abbreviation Dict
    abbr_dict = {}
    for i, row in abbr.iterrows():
        abbr_dict[row['Word']] = row['abbr']
    abbr_dict
    
    clean_text = []
    
    for token in text:
        # Further check for stop words
        if (token in custom_stopwords) or ('asu.edu' in token) or ('yiv' in token):
            continue
        # Replace abbreviations
        if token in abbr_list:
            token = abbr_dict[token]

        clean_text.append(token)
    
    tmp = (" ").join(clean_text)
    
    # Replacing fasfa with fafsa
    tmp =  tmp.replace('fasfa', 'fafsa')
    tmp =  tmp.replace('refunded', 'refund')
    tmp = tmp.replace('#', '')
    return tmp


def replace_with_mappings(text):
    word_map = pd.read_csv(supporting_files_path+'word_mapping.csv')
    map_dict = dict(zip(word_map.word, word_map.mapping))
    
    clean_text = []
    
    for token in text:

        # Replace with mapped words
        if token in map_dict.keys():
            token = map_dict[token]
        clean_text.append(token)
      
    tmp = (" ").join(clean_text)
    tmp = tmp.replace('#', '')
    return clean_text


def preprocess(df, replace_col='comments', initial=True):
    replies_regex = None
    forwards_regex = None
    with open(supporting_files_path+'replies_regex.txt') as myfile:
        replies_regex = r"{}".format(myfile.readlines()[0])
    with open(supporting_files_path+'forwards_regex.txt') as myfile:
        forwards_regex = r"{}".format(myfile.readlines()[0])

    if initial:
        df['Initial_conv'] = df[replace_col].apply(lambda doc: re.split(replies_regex + '|' + forwards_regex, str(doc), flags=re.IGNORECASE)[-1])

        df['Final_conv'] = df[replace_col].apply(lambda doc: re.split(replies_regex + '|' + forwards_regex, str(doc), flags=re.IGNORECASE)[0])

        replace_col = 'Initial_conv'

    logger.info('Starting preprocess')
    logger.info('Replacing N-grams...')
    final_col = replace_col + '_preprocessed'
    # N gram processing
    global total
    total = len(df)
    df[final_col] = df[replace_col].apply(replace_ngrams)

    logger.info('General Preprocessing...')
    # Preprocessing
    df[final_col] = df[final_col].apply(text_preprocessing)

    logging.info('Final Preprocessing...')
    # Final preprocessing
    df[final_col] = df[final_col].apply(replace_with_mappings)
    df[final_col] = df[final_col].apply(join_wrds)
    df[final_col] = df[final_col].apply(lambda x: re.sub('aa+', '', x))
    
    return df
