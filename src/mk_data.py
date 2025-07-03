import re
import pandas as pd
import json
import tqdm
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
#from langchain.vectorstores import Chroma


def normalize_text(s, sep_token = " \n "):
    '''
    データ整形用の関数を定義
    '''
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()

    return s


def str_to_lst(data):
        data["pos"] = [data["pos"]]
        return data

    
def add_negative_samples(df, neg_num=10):
    """
    DataFrameに対してネガティブサンプリングを行い、negカラムを追加
    
    Parameters:
    - df: 元のDataFrame
    - neg_num: ネガティブサンプルの数
    
    Returns:
    - 新しいDataFrame（negカラム付き）
    """
    df_result = df.copy()
    new_col = []
    
    for i in range(len(df_result)):
        # ランダムなインデックスを生成（自分自身を除く）
        ids = np.random.randint(0, len(df_result), size=neg_num)
        while i in ids:
            ids = np.random.randint(0, len(df_result), size=neg_num)
        
        # ネガティブサンプルを取得
        neg = [df_result.iloc[idx]["pos"] for idx in ids]
        new_col.append(neg)
    
    # negカラムを追加
    df_result["neg"] = new_col

    return df_result

    
def main (tgt_file):
    
    df_orig = pd.read_json(tgt_file, orient='records', lines=True)    
    #print(df)

    # データ格納用に空のDataFrameを作成
    df = pd.DataFrame()
    df['query'] = df_orig['title']
    df['pos'] = df_orig['description']
    #df['doc_id'] = df_orig['doc_id']
    df['item_id'] = df_orig['item_id']
    df['id'] = range(1, len(df) + 1) 

    np.random.seed(42)
    neg_num = 10

    # ネガティブサンプルを追加
    df = add_negative_samples(df, neg_num=neg_num)
    
    # posカラムをリスト形式に変換
    df = df.apply(str_to_lst, axis=1)

    instruction = "Represent this sentence for searching relevant passages: "

    # DataFrameに新しいカラム"prompt"を追加
    df["prompt"] = instruction
    
    df.to_json('prueba.jsonl', orient='records', force_ascii=False, lines=True)

    d_train, d_test = train_test_split(df, test_size=0.1, shuffle=True, random_state=1) #80% 20%に分割
    d_train.to_json('prueba_train.jsonl', orient='records', force_ascii=False, lines=True)
    d_test.to_json('prueba_test.jsonl', orient='records', force_ascii=False, lines=True)

    dqueries = pd.DataFrame()
    dqueries['text'] = d_test['query']
    dqueries['id'] = d_test['id']

    dcorpus = pd.DataFrame()
    dcorpus['text'] = df['pos']
    dcorpus['id'] = df['id']

    dqrels = pd.DataFrame()    
    dqrels['qid'] = d_test["id"]
    dqrels["docid"] = d_test["id"].tolist()  # docidカラムを追加
    dqrels["relevance"] = 1  # relevanceカラムを追加（全て1）

    dqueries.to_json('prueba_queries.jsonl', orient='records', force_ascii=False, lines=True)
    dcorpus.to_json('prueba_corpus.jsonl', orient='records', force_ascii=False, lines=True)
    dqrels.to_json('prueba_qrels.jsonl', orient='records', force_ascii=False, lines=True)    

    
            
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True)    # Target Data_Dir
    args = parser.parse_args()
    tgtfile = args.file
    
    main(tgtfile)

