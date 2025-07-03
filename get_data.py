import re
import pandas as pd
import json
import tqdm
import argparse
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


def main (tgt_file):

    info = get_data(tgt_file)
    # DataFrame を作成
    df = pd.DataFrame(info)
    #print(df)

    df.to_json('rm_documents.jsonl', orient='records', force_ascii=False, lines=True)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100
    )

    # データ格納用に空のDataFrameを作成
    df_out = pd.DataFrame(columns=["doc_id", "chunk_no", "content"])

    for i in tqdm.tqdm(range(0, len(df.index))):
        title = str(df.loc[i, 'title'])
        desc = str(df.loc[i, 'description'])
        docid = str(df.loc[i, 'document_id'])
        text = title + '\n' + desc
        
        chunk = text_splitter.split_text(text)
        for i, c in enumerate(chunk):
            df_out = pd.concat([df_out, pd.DataFrame({"doc_id": docid, "chunk_no": i, "content": c}, index=[0])])

    # データ識別用にIDを作成
    df_out["ID"] = df_out["doc_id"].str.replace(".txt", "", regex=False) + "〓" + df_out["chunk_no"].astype(str)

    df_out = df_out.reset_index(drop=True)

    # データ整形
    df_out["content"]= df_out["content"].apply(lambda x : normalize_text(x))    

    df_out.to_json('rm_split_documents.jsonl', orient='records', force_ascii=False, lines=True)

    
def get_data (tgt_file):
    '''
    CXDからもらっている元のJSONLデータから、必要な項目の情報を抽出する
    '''
    info = dict()
    with open(tgt_file) as f:
        #json_dic = json.load(f)
        jsonl_data = [json.loads(l) for l in f.readlines()]
        for json_line in jsonl_data:

            document_id = json_line['document_id']
            print (f"document_id : {document_id}")
            title = json_line['title']
            print (f"title : {title}")
            description = json_line['description']['published']

            length_desc = len(description)
            print (f"len_desc : {length_desc}")
            
            info.setdefault('document_id', []).append(document_id)
            info.setdefault('title', []).append(title)
            info.setdefault('description', []).append(description)
            
    return info


            
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True)    # Target Data_Dir
    args = parser.parse_args()
    tgtfile = args.file
    
    main(tgtfile)

