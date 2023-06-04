# 데이터 중심의 주제 분류 대회 (Data-centric Topic Classification Competition)
boostcamp AI Tech 5 NLP 트랙 레벨2 두 번째 프로젝트

모델 구조의 변경 없이 Data-Centric 관점으로 텍스트 주제 분류하기


## 일정 Schedule
프로젝트 전체 기간(7일+): 5월 24일 (수) 10:00 ~ 6월 1일 (목) 19:00


## 팀 Team
|문지혜|박경택|박지은|송인서|윤지환|
|:---:|:---:|:---:|:---:|:---:|
|<img src="https://avatars.githubusercontent.com/u/85336141?v=4" width="120" height="120">|<img src="https://avatars.githubusercontent.com/u/97149910?v=4" width="120" height="120">|<img src="https://avatars.githubusercontent.com/u/97666193?v=4" width="120" height="120">|<img src="https://avatars.githubusercontent.com/u/41552919?v=4" width="120" height="120">|<img src="https://avatars.githubusercontent.com/u/37128004?v=4" width="120" height="120">|
|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:munjh1121@gmail.com)](mailto:afterthougt@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:afterthougt@gmail.com)](mailto:afterthougt@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:imhappyhill@gmail.com)](mailto:imhappyhill@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:songinseo0910@gmail.com)](mailto:songinseo0910@gmail.com)|[![Gmail Badge](https://img.shields.io/badge/Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:yjh091500@naver.com)](mailto:yjh091500@naver.com)|
|[![GitHub Badge](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github&link=https://github.com/jihye-moon)](https://github.com/jihye-moon)|[![GitHub Badge](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github&link=https://github.com/afterthougt)](https://github.com/afterthougt)|[![GitHub Badge](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github&link=https://github.com/iamzieun)](https://github.com/iamzieun)|[![GitHub Badge](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github&link=https://github.com/fortunetiger)](https://github.com/fortunetiger)|[![GitHub Badge](https://img.shields.io/badge/-GitHub-black?style=flat-square&logo=github&link=https://github.com/ohilikeit)](https://github.com/ohilikeit)|


## 프로젝트 갈무리 보고서 Wrap-up Report
[Wrap-up Report](https://github.com/boostcampaitech5/level2_nlp_datacentric-nlp-12/blob/main/docs/boostcamp_ai_tech_5th_nlp_level2_datacentric_wrapup_report_team_12.pdf)


## 저장소 구조 Repository Structure
```
level2_nlp_datacentric-nlp-12/
├── docs/                                         // 프로젝트 갈무리 보고서 등
│
├── notebook/                                     // 데이터 증강 및 클리닝, 사후 분석
│   ├── data_augmentation/
│   │   ├── easy_data_augmentation.ipynb          // EDA (Easy Data Augmentation) 증강 실험
│   │   └── hub_data.ipynb                        // AI Hub 데이터 증강 실험
│   │
│   ├── data_cleaning/
│   │   ├── data_cleaner_g2p_crawling.ipynb       // 웹 스크레이핑을 통한 G2P 제거 실험
│   │   ├── data_cleaner_labeling_error.ipynb     // 레이블 오류 보강 실험
│   │   ├── data_cleaner_cleanlab.ipynb           // cleanlab 라이브러리 활용 데이터 클리닝 실험
│   │   ├── data_cleaner_seq2seq_denoising.ipynb  // Seq2Seq Denoising 모델 사용 데이터 클리닝 실험
│   │   └── ...
│   │
│   └── post_analysis/
│
├── src/
│   ├── back_translate.py                         // 역번역
│   ├── config.yaml                               // 데이터 등 실험 설정 변경
│   ├── data_cleaner_with_gpt.py                  // GPT-3.5-turbo를 사용한 데이터 클리닝
│   ├── dataset.py                                // 커스텀 데이터셋 정의
│   ├── preprocess.py                             // 데이터 전처리
│   ├── run.py                                    // 실험 수행
│   ├── tokenization_kobert.py                    // KoBERT 토크나이저
│   └── util.py                                   // 기타 유틸리티 함수
│
├── README.md
└── requirements.txt
```

## 사용법 Usage
- 환경 설치

```bash
pip install -qU -r requirements.txt
```

- 기본 설정으로 학습 및 추론

```bash
python src/run.py
```

- 역번역 (webdriver 설치가 가능한 환경에서)

```bash
python src/back_translate.py
```

- GPT-3.5-turbo를 사용한 데이터 클리닝

```bash
python src/data_cleaner_with_gpt.py --openai_api_key "OpenAI에서 발급 받은 API 키 값"
```


## 데이터 Data
KLUE(Korean NLU Benchmark) 주제 분류(topic classification) 데이터셋 일부에 노이즈 처리를 한 데이터셋
- train: 31,974개
- dev: 13,704개
- test: 9,107개

데이터 주제 분류(target 칼럼)
- 0: IT과학
- 1: 경제
- 2: 사회
- 3: 생활문화
- 4: 세계
- 5: 스포츠
- 6: 정치 

데이터 예시

| ID | text | target | url | date |
| --- | --- | --- | --- | --- |
| ynat-v1_train_08697 | 배구 남자대표팀 감독 공모에 임도헌 코치 단독 지원 | 5 | https://sports.news.naver.com/news.nhn?oid=001... | 2019.05.24 17:17 |


## 평가 방법 Evaluation Metric
Macro F1 점수로 평가


## 시도한 방법 및 결과 Trials and Results
**자세한 내용 및 해석, 고찰 등은 위 보고서 참고**

### 데이터 전처리

| 실험명 | 예시 | Validation F1 | Public LB (leaderboard) F1 |
| --- | --- | --- | --- |
| baseline | 묘비명 알리…故무하마드 알리 10만명 추모받으며 영면종합 | 0.8371 | 0.8738 |
| 한자 제거 | 묘비명 알리…무하마드 알리 10만명 추모받으며 영면종합 | 0.8265 |  |
| 한자 to 한글 | 묘비명 알리…고무하마드 알리 10만명 추모받으며 영면종합 | **0.839** | 0.8685 |
| 특수문자 제거 | 묘비명 알리故무하마드 알리 10만명 추모받으며 영면종합 | 0.8365 | 0.8590 |
| 특수문자 to 공백 | 묘비명 알리 故무하마드 알리 10만명 추모받으며 영면종합 | **0.8379** |  |
| ‘종합~’ 제거 | 묘비명 알리…故무하마드 알리 10만명 추모받으며 영면 | 0.8367 |  |

### 데이터 클리닝
#### G2P 에러 제거
GPT-3.5-turbo 사용 결과
- 실험군: baseline 데이터
- 대조군 1: baseline 데이터의 일부 행 교체(‘사회’ 분류 데이터에 대해서만 데이터 노이즈 교정한 데이터)
- 대조군 2: baseline 데이터 + 증강 데이터(‘사회’ 분류 데이터에 대해서만 데이터 노이즈 교정한 데이터)

|  | Validation F1 | Public LB F1 | Private LB F1 |
| --- | --- | --- | --- |
| 실험군 | **0.838** | **0.8729** | **0.8597** |
| 대조군 1 | 0.8278 | 0.8675 | 0.8537 |
| 대조군 2 | 0.8317 | 0.8603 | 0.8418 |

#### 레이블링 에러 제거
Cleanlab 라이브러리 사용 결과
|  | 데이터 설명 | Public LB F1 | Private LB F1 | Validation F1 |
| --- | --- | --- | --- | --- |
| 원본 데이터 | 라벨링 에러 있음, G2P noise 있음 | - | - | 0.8371 |
| (baseline) G2P 노이즈 제거 후 데이터 | 라벨링 에러 있음 | - | - | 0.8455 |
| 라벨링 에러 제거 후 데이터 version 1 | 약 1200개의 라벨링 오류가 있는 데이터를 대체함 | **0.8798** | 0.8542 | 0.8632 |
| 라벨링 에러 제거 후 데이터 version 2 | 1,310개의 라벨링 오류가 있는 데이터를 대체함 | 0.8750 | **0.8658** | **0.8750** |

### 데이터 증강
#### EDA (Easy Data Augmentation)
koeda 라이브러리로 random deletion 및 random swap 방법을 사용한 데이터 증강 결과

| 데이터 설명 | Public LB F1 | Validation F1 |
| --- | --- | --- |
| Baseline | **0.8798** | **0.8632** |
| RD  training dataset만 10% 증강 | 0.8687 | 0.8611 |
| RS  training dataset만 10% 증강 | 0.8675 | 0.8616 |

#### 공개 데이터 사용 데이터 증강
AI Hub 공개 데이터를 사용한 데이터 증강 결과

| 번호 | 데이터 설명 | Public LB F1 | Validation | 비고 |
| --- | --- | --- | --- | --- |
| 1 | 기본(baseline) | **0.8769** | 0.8371 |  |
| 2 | 기본 + ai-hub(89,000개) | 0.8441 | 0.8056 | 가장 많은 사회라벨을 2만개 랜덤하게 없앴다.  |
| 3 | 기본 + ai-hub(9,000개) | 0.8545 | 0.8258 | [1500, 1500, 1500, 1500, 1000, 1000, 1000] 비율로 추가 |
| 4 | 기본 + ai-hub(9000개, cleanlab 기준 label_quality 높은 순서대로) | 0.8705 | 0.85766 | [1500, 1500, 1500, 1500, 1000, 1000, 1000] 비율로 추가 |
| 5 | 기본 + ai-hub 사회 1500개만 추가 | 0.8768 | **0.8624** | cleanlab 기준 label_quality 높은 순서대로 |

#### 역번역(Back translation) 증강
네이버 파파고 번역기를 사용한 train 데이터 증강 결과

|  | Validation F1 at last epoch | Public LB F1 | Private LB F1 |
| --- | --- | --- | --- |
| 기본 (대조군) | **0.838** | **0.8729** | 0.8597 |
| 역번역 대체 (실험군 1) | 0.8266 | 0.8602 | 0.8495 |
| 기본 + 역번역 증강 (실험군 2) | 0.8352 | 0.8709 | **0.8599** |

