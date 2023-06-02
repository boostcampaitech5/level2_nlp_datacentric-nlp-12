import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def prepare_df(PATH):
    """
    주어진 파일 경로(PATH)로부터 csv 파일을 불러와 데이터프레임 형태로 반환합니다.
    
    Parameters:
        PATH (str): 데이터 파일 경로
        
    Returns:
        df (pd.DataFrame): 변환된 데이터프레임
    """
    df = pd.read_csv(PATH)

    mapping_dict = {0: "IT과학", 1: "경제",  2: "사회", 3: "생활문화", 
                    4: "세계", 5: "스포츠", 6: "정치"}

    df['target'] = df['target'].map(mapping_dict)
    df['pred'] = df['pred'].map(mapping_dict)

    return df


def compare_df(df_new, df_base):
    """
    두 개의 데이터프레임(df_new, df_base)을 병합한 데이터프레임을 반환합니다.
    
    Parameters:
        df_new (pd.DataFrame): 비교할 새로운 데이터프레임
        df_base (pd.DataFrame): 비교할 기존 데이터프레임
        
    Returns:
        df (pd.DataFrame): 비교 및 병합된 데이터프레임
    """
    df_new = df_new.rename(columns = {'text': 'text_new', 'pred': 'pred_new'})
    df_base = df_base.rename(columns = {'text': 'text_base', 'pred': 'pred_base'})
    df = pd.merge(df_new, df_base, on=['ID', 'target', 'url', 'date'])
    df = df[['ID', 'text_new', 'text_base', 'target', 'pred_new', 'pred_base']]
    
    return df


def cm_graph(df: pd.DataFrame, correct:bool = True):
    """
    주어진 데이터프레임(df)의 'target'과 'pred' 열을 사용하여 confusion matrix을 계산하고
    heatmap 형태로 시각화합니다.

    arguments:
        df (pd.DataFrame): 'target'과 'pred' 열이 포함된 데이터프레임
    
    return:
        None. 함수는 confusion matrix heatmap을 출력합니다.
    """
    if correct == False:
        df = df[df['target'] != df['pred']]

    # confusion matrix 계산
    cm = confusion_matrix(df['target'], df['pred'])

    # 라벨 설정
    labels = sorted(list(df['target'].unique()))

    # heatmap 그리기
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, annot_kws={"size":8}, fmt='g', xticklabels=labels, yticklabels=labels, cmap='OrRd')

    # 축 이름 및 제목 설정
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # 그래프 표시
    plt.show()


def cm_dataframe(df: pd.DataFrame, sort_column: str = 'label') -> pd.DataFrame:
    """
    주어진 데이터프레임(df)의 'target'과 'pred' 열을 사용하여 
    각 label에 대한 confusion matrix을 계산하고 dataframe 형태로 시각화합니다.

    arguments:
        df (pd.DataFrame): 'target'과 'pred' 열이 포함된 데이터프레임
        sort_column (str): confusion matrix dataframe을 정렬하는 기준 열
    
    return:
        metric_df (pd.DataFrame): confusion matrix dataframe
    """
    label_list = list(sorted(df['target'].unique()))

    target = [len(df[df['target'] == label]) for label in label_list]
    pred = [len(df[df['pred'] == label]) for label in label_list]
    TP = [len(df[(df['pred'] == label) & (df['target'] == label)]) for label in label_list]
    FP = [len(df[(df['pred'] == label) & (df['target'] != label)]) for label in label_list]
    FN = [len(df[(df['pred'] != label) & (df['target'] == label)]) for label in label_list]

    precision = []
    for tp, fp in zip(TP, FP):
        if tp + fp > 0:
            p = round(tp / (tp + fp), 4) 
        else:
            p = 0
        precision.append(p)

    recall = []
    for tp, fn in zip(TP, FN):
        if tp + fn > 0:
            r = round(tp / (tp + fn), 4)
        else:
            r = 0
        recall.append(r)
    
    f1_score = []
    for p, r in zip(precision, recall):
        f1 = round(2 * p * r / (p + r) , 4)
        f1_score.append(f1)

    metric_df = pd.DataFrame(zip(label_list, target, pred, TP, FP, FN, precision, recall, f1_score))
    metric_df.columns = ['label', 'target', 'pred', 'TP', 'FP', 'FN', 'precision', 'recall', 'f1 score']
    metric_df = metric_df.sort_values(sort_column)

    return metric_df


def total_metric(df: pd.DataFrame) -> pd.DataFrame:
    """
    주어진 데이터프레임(df)의 'target'과 'pred' 열을 사용하여 
    전체 데이터에 대한 confusion matrix을 계산하고 dataframe 형태로 시각화합니다.

    arguments:
        df (pd.DataFrame): 'target'과 'pred' 열이 포함된 데이터프레임
    
    return:
        metric_df (pd.DataFrame): 주어진 데이터에 대한 confusion matrix dataframe
    """
    df = cm_dataframe(df)
    
    precision = round(sum(df['precision']) / 7, 4)
    recall = round(sum(df['recall']) / 7, 4)
    F1 = round(sum(df['f1 score']) / 7, 4)

    metric_dict = {"precision": precision, 
                   "recall": recall, 
                   "f1 score": F1}
    metric_df = pd.DataFrame.from_dict(data = metric_dict, 
                                        orient='index', 
                                        columns=['value'])

    return metric_df


def precision_recall_graph(df: pd.DataFrame):
    """
    주어진 데이터프레임(df)의 'target'과 'pred' 열을 사용하여 
    각 label에 대한 precision과 recall을 계산하고 scatterplot 형태로 시각화합니다.

    arguments:
        df (pd.DataFrame): 'target'과 'pred' 열이 포함된 데이터프레임
    
    return:
        None. 함수는 precision과 recall에 대한 scatterplot을 출력합니다.
    """
    label_list = list(sorted(df['target'].unique()))

    TP = [len(df[(df['pred'] == label) & (df['target'] == label)]) for label in label_list]
    FP = [len(df[(df['pred'] == label) & (df['target'] != label)]) for label in label_list]
    FN = [len(df[(df['pred'] != label) & (df['target'] == label)]) for label in label_list]

    precision = []
    for tp, fp in zip(TP, FP):
        if tp + fp > 0:
            p = round(tp / (tp + fp), 4) 
        else:
            p = 0
        precision.append(p)

    recall = []
    for tp, fn in zip(TP, FN):
        if tp + fn > 0:
            r = round(tp / (tp + fn), 4)
        else:
            r = 0
        recall.append(r)

    plt.figure(figsize=(5, 3))
    for i, label in enumerate(label_list):
        plt.scatter(recall[i], precision[i], label=label)
    plt.legend()


    # 그래프 제목과 축 레이블 설정
    plt.title('relation between recall and precision')
    plt.xlabel('recall')
    plt.ylabel('precision')

    # 그래프 보이기
    plt.show()


def compare_matrix(df_total):
    """
    주어진 데이터프레임(df_total)을 기반으로 개선 및 악화된 경우의 수를 비교하여 결과를 데이터프레임으로 반환합니다.

    Parameters:
        df_total (pandas.DataFrame): 비교할 데이터프레임

    Returns:
        matrix_df (pandas.DataFrame): 개선 및 악화된 경우의 수를 포함한 결과 데이터프레임
    """
    improved = len(df_total[(df_total['target'] == df_total['pred_new']) & (df_total['target'] != df_total['pred_base'])])
    worsen = len(df_total[(df_total['target'] != df_total['pred_new']) & (df_total['target'] == df_total['pred_base'])])
    both_right = len(df_total[(df_total['target'] == df_total['pred_new']) & (df_total['target'] == df_total['pred_base'])])
    both_wrong = len(df_total[(df_total['target'] != df_total['pred_new']) & (df_total['target'] != df_total['pred_base'])])

    compare_dict = {"improved": improved, "worsen": worsen, "both right": both_right, "both wrong": both_wrong}
    matrix_df = pd.DataFrame.from_dict(data = compare_dict,
                                       orient='index', 
                                       columns=['value'])

    return matrix_df