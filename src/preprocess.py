import pandas as pd
import re

import hanja


def preprocess_df(df: pd.DataFrame, hanja: bool = False, special: bool = False, jonghab: bool = False) -> pd.DataFrame:
    
    # 한자 한국어로 대체
    if hanja == True:
        df['text'] = [hanja.translate(text, 'substitution') for text in df['text']]

    # 특수문자 제거
    if special == True:
        df['text'] = [re.sub(r'[^\w\s\u4E00-\u9FFF]', ' ', text) for text in df['text']]

    # '종합 ~' 제거
    if jonghab == True:
        pattern = re.compile(r'(.*)\s종합$')
        df['text'] = df['text'].apply(lambda x: re.sub(pattern, r'\1', x))

    return df