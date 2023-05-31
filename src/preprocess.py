import pandas as pd
import re

import hanja


def preprocess_df(df: pd.DataFrame, hanja: bool = False, special: bool = False, jonghab: bool = False) -> pd.DataFrame:
    """
    주어진 데이터프레임을 전처리합니다.

    Parameters:
        df (pd.DataFrame): 전처리할 데이터프레임.
        hanja (bool, optional): 한자를 한글로 대체할지 여부를 나타내는 인자. 기본값은 False.
        special (bool, optional): 특수문자 제거 여부를 나타내는 인자. 기본값은 False.
        jonghab (bool, optional): 문장 마지막 어절의 '종합 ~' 패턴 제거 여부를 나타내는 인자. 기본값은 False.

    Returns:
        pd.DataFrame: 전처리가 적용된 데이터프레임.

    """
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