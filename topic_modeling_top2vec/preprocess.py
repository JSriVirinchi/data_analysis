import pandas as pd
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import imp
import preprocessing
imp.reload(preprocessing)
from preprocessing import preprocess


def replace_words(x, words):
    for i in words:
        x = x.replace(i,'')
    return x

input_file = '../steps/email_training_input_data.csv'
df = pd.read_csv(input_file)
df = df[df['description'].notna()]
df['description'] = df['description'].apply(lambda x: x.lower())

preprocessed_df = preprocess(df)

rmv_wrd = ['&nbsp', 'nbsp', 'px ', 'px;', 'px!', 'px)', 'font', 'span', 'margin', 'thead', 'borderright', 'border', 'header', 'overflow', 'mso', 'width:', 'div,', 'div{',
           'size:', 'height:', 'weight', 'underline', 'yshortcut', 'size ', 'height ', 'alt', 'alt:', 'weight ', 'weight:', 'arial', 'font', 'width ', 'padding:', 'padding ',
           ' verdana ', 'asu?s', ' unsub ', ' subscription ', ' unsubscribe ', ' sincerely ', ' significantly ']

preprocessed_df['Initial_conv_preprocessed'] = preprocessed_df['Initial_conv_preprocessed'].apply(lambda x: replace_words(x, rmv_wrd))
preprocessed_df.to_csv('../steps/email_trainining_input_data_preprocessed.csv', index=False)