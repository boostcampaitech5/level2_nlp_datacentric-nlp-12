import pandas as pd
import re


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    
    # 한자 제거
    #df['text'] = [re.sub('[一-龥]', '', text) for text in df['text']]
    
    # 특수문자 제거
    df['text'] = [re.sub(r'[^\w\s\u4E00-\u9FFF]', '', text) for text in df['text']]

    return df