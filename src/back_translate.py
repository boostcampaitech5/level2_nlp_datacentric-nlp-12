import argparse
import time
import urllib.parse
from functools import partial
from typing import Any, List, Tuple

import pyderman
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()


def apply_bt_on_dataframe(row: pd.core.series.Series,
                          driver: Any,
                          target_present: Any,
                          tgt_cols: List[str] = ['text'])\
        -> pd.core.series.Series:
    """데이터프레임의 인덱스를 받아 타깃 칼럼에 대하여 역번역 적용.

    데이터프레임의 인덱스에서 iterable 형태로 정의된 타깃 칼럼 tgt_cols에 대해
    papago_back_translate 함수 적용.

    Args:
        row: 처리할 데이터의 행. Pandas의 DataFrame에서 추출한 1개 행으로서
            필요 정보 포함.
        driver: 웹 스크래핑이나 자동화 작업을 수행하기 위해 사용되는 웹 드라이버.
            보통 Selenium의 WebDriver 인스턴스를 지칭.
        target_present: 웹 페이지에서 특정 요소의 존재 여부를 확인하는 데 사용되는 locator.
            웹 페이지의 구조를 파악하고 특정 요소가 존재하는지 확인하는데 사용.
            이 함수에서는 '//*[@id="txtTarget"]'라는 XPATH를 갖는 요소의 존재 여부를 확인.
        tgt_cols: 역번역을 적용할 칼럼(들).
        
    Returns:
        타깃 칼럼에 대하여 역번역이 적용된 Pandas Series 타입 인덱스.
    """

    for tgt_col in tgt_cols:
        src_text = row[tgt_col]
        _, tgt_text = papago_back_translate(driver, target_present, src_text)
        row[tgt_col] = tgt_text
    return row.to_frame().T


def papago_back_translate(driver: Any,
                          target_present: Any,
                          src_text: str,
                          src_lang: str = 'ko',
                          via_lang: str = 'en',
                          tgt_lang: str = 'ko')\
        -> Tuple[str, str]:
    """주어진 텍스트를 중간 언어(via_lang)를 거쳐 타깃 언어(tgt_lang)로 역번역.

    Args:
        driver: 웹 스크래핑이나 자동화 작업을 수행하기 위해 사용되는 웹 드라이버.
            보통 Selenium의 WebDriver 인스턴스를 지칭.
        target_present: 웹 페이지에서 특정 요소의 존재 여부를 확인하는 데 사용되는 locator.
            웹 페이지의 구조를 파악하고 특정 요소가 존재하는지 확인하는데 사용.
            이 함수에서는 '//*[@id="txtTarget"]'라는 XPATH를 갖는 요소의 존재 여부를 확인.
        src_text: 주어진 텍스트.
        src_lang: 주어진 텍스트의 언어. 기본값: 'ko' (한국어)
        via_lang: 중간 번역 언어. 기본값: 'en' (영어)
        tgt_lang: 타깃 번역 언어. 기본값: 'ko' (한국어)

    Returns:
        via_text: 중간 번역 결과 텍스트.
        tgt_text: 타깃 번역 결과 텍스트.
    """

    try:
        # ^과 같은 특수문자는 url에 직접 삽입이 불가하여 인코딩 필요
        # urllib.parse.quote 함수로 url 인코딩 수행 가능
        encoded_src_text = urllib.parse.quote(src_text)
        # from source to intermediate
        driver.get(f'https://papago.naver.com/?'
                   f'sk={src_lang}&tk={via_lang}&st={encoded_src_text}')
        element = WebDriverWait(driver, 1.2).until(target_present)
        driver.implicitly_wait(1.5)
        time.sleep(1.5)
        via_text = element.text

        # url 인코딩
        encoded_via_text = urllib.parse.quote(via_text)
        driver.get(f'https://papago.naver.com/?'
                   f'sk={via_lang}&tk={tgt_lang}&st={encoded_via_text}')
        element = WebDriverWait(driver, 1.2).until(target_present)
        driver.implicitly_wait(1.5)
        time.sleep(1.5)
        tgt_text = element.text
        return via_text, tgt_text
    except TimeoutException:
        return 'Back translation failed', '역번역 실패'


def back_translate(driver, target_present, args):
    df = pd.read_csv(args.input_csv)

    # functools.partial을 사용하여 apply_bt_on_dataframe 함수의 일부 인자 고정
    func = partial(apply_bt_on_dataframe,
                   driver=driver,
                   target_present=target_present,
                   tgt_cols=args.tgt_cols,
                   save_path=args.save_dir)
    
    batch_size = 10
    rows = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if i == 0:
            rows.append(func(row, first_row=True))  # 첫 번째 행에 대해서만 header를 True로 설정
        else:
            rows.append(func(row))
        
        if (i+1) % batch_size == 0:
            pd.concat(rows).to_csv(args.save_dir, mode='a', header=(i < batch_size), index=False)
            rows = []

    # 마지막 배치를 저장합니다.
    if rows:
        pd.concat(rows).to_csv(args.save_dir, mode='a', header=False, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, default='../data/train.csv', help='Input CSV file')
    parser.add_argument('--tgt_cols', nargs='+', type=str, default=['text'], help='Target columns for back translation')
    parser.add_argument('--via_lang', type=str, default='en', help='Intermediate language for back translation (default: en)')
    parser.add_argument('--save_dir', type=str, default='../data/train_bt.csv', help='Output path for the back-translated csv file')
    args = parser.parse_args()

    driver_path = pyderman.install(browser=pyderman.chrome)
    print(f'Installed geckodriver driver to path: {driver_path}')

    service = Service(executable_path=driver_path)
    driver = webdriver.Chrome(service=service)
    driver.implicitly_wait(2)
    time.sleep(2)

    target_present = EC.presence_of_element_located((By.XPATH, '//*[@id="txtTarget"]'))

    back_translate(driver, target_present, args)


if __name__ == '__main__':
    main()
