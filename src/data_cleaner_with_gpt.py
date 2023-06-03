import argparse
import json

import openai
import pandas as pd
from langchain.chat_models import ChatOpenAI
from tqdm import tqdm


def clean_data(api_key, data_path, column_name, chunk_size, target) -> None:
    """GPT-3.5-turbo를 사용한 데이터 클리너.

    Args:
        api_key (str): OpenAI API call을 위해 필요한 키 값.
            기본값 'YOUR_API_KEY'.
        data_path (str): 데이터 클리닝을 수행할 데이터프레임의 경로.
            기본값 '../data/train.csv'.
        column_name (str): 위 데이터프레임에 대해 클리닝을 수행할 칼럼명.
            기본값 'text'.
        chunk_size (int): 번역을 수행할 한계 단위 문장 수.
            기본값 40.
        target (int): 타깃 주제 레이블. 0부터 6까지의 정수 값을 가짐.
            0 - IT과학, 1 - 경제, 2 - 사회, 3 - 생활문화, 4 - 세계, 5 - 스포츠, 6 - 정치.
            기본값 2.
    """
    df_train = pd.read_csv(data_path)

    df_filtered = df_train[df_train['target'] == target]
    df_filtered = df_filtered.reset_index()

    df_filtered_to_save = df_filtered.copy()
    df_filtered_to_save.loc[:, 'text'] = ''
    
    num_to_topic = {
        0: 'it_and_science',
        1: 'economy',
        2: 'society',
        3: 'life_and_culture',
        4: 'world',
        5: 'sports',
        6: 'politics',
    }

    prompt = """아래 각 text에 대해 노이즈가 포함되어 있다고 판단되면 원 문장으로 수정해줘.
이때, 아래에 있는 rules를 만족해야만 해. Examples를 참고해서 문장을 잘 가다듬어 줘.

[Texts]
{}

[Rules]
1. 어법 및 어문 규정을 준수할 것
2. 노이즈가 포함된 문장은 수정될 원 문장과 어절 및 음절 수가 최대한 같을 것
3. 알맞은 문맥을 추론할 것
4. 구개음화, 된소리 되기와 같은 현상이 일어난 노이즈 문장이 많음을 인지할 것
5. '종합'과 같은 단어가 문장 끝에 붙은 경우, 의미가 없을 확률이 높으니 생략할 것
6. 불필요한 특수문자는 최대한 생략하며, 문장 끝에 마침표를 생략할 것

[Example 1]
(수정 전) 방통시믜위 불법 때부업 쩡보 등 오백꾸시보건 삭쩨 등 요구
(수정 후) 방통심의위 불법 대부업 정보 등 595(오백구십오)건 삭제 등 요구

[Example 2]
(수정 전) 여수 윤화류 보관 창고 화재현장서 치손는 거므 년기	
(수정 후) 여수 윤활유 보관 창고 화재 현장서 치솟는 검은 연기

Provide them in JSON format with the following key: 
index and text.
"""

    total_rows = len(df_filtered)
    for i in tqdm(range(0, total_rows, chunk_size)):
        if i + chunk_size > total_rows:
            chunk = df_filtered[i:]
        else:
            chunk = df_filtered[i:i+chunk_size]

        texts = chunk[['index', column_name]].apply(lambda row: f'{row["index"]},{row[column_name]}\n', axis=1).str.cat()
        full_prompt = prompt.format(texts)

        chat_open_ai = ChatOpenAI(openai_api_key=api_key, temperature=0.1)

        success = False
        while not success:
            try:
                print(full_prompt)
                response_chat_open_ai = chat_open_ai.predict(full_prompt)
                json_data = json.loads(response_chat_open_ai)
                for item in json_data:
                    index, text = item['index'], item[column_name]
                    df_filtered_to_save.loc[df_filtered_to_save['index'] == index, column_name] = text
                df_filtered_to_save.to_csv(f'../data/train_{target}_{num_to_topic[target]}.csv', index=False)
                success = True
            except openai.error.AuthenticationError:
                print('Error: Invalid or missing OpenAI API key. Please provide a valid key.')
                return
            except json.JSONDecodeError:
                print(f'Failed to clean chunk {i} due to json.JSONDecodeError. Retrying...')
                continue


def main():
    parser = argparse.ArgumentParser(description='Clean data using GPT-3.5-turbo')
    parser.add_argument('--openai_api_key', type=str, default='YOUR_API_KEY',
                        help='API key for OpenAI')
    parser.add_argument('--data_path', type=str, default='../data/train.csv',
                        help='Path to the data file')
    parser.add_argument('--column_name', type=str, default='text',
                        help='Name of the column to be transformed')
    parser.add_argument('--chunk_size', type=int, default=40,
                        help='Chunk size for processing')
    parser.add_argument('--target', type=int, default=2,
                        help='Target value for filtering data')

    args = parser.parse_args()

    clean_data(args.openai_api_key, args.data_path, args.column_name, args.chunk_size, args.target)


if __name__ == '__main__':
    main()
