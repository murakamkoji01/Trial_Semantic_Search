import re
import pandas as pd
import json
import tqdm
import argparse
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
#from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

import chromadb.utils.embedding_functions as embedding_functions
from chromadb.utils.batch_utils import create_batches

#from langchain.embeddings import OpenAIEmbeddings
#from langchain.vectorstores import Chroma



def main (tgt_file):

    df = pd.read_json(tgt_file, orient='records', lines=True)    

    # contentとmetadataをリスト型で定義
    content_list = df["content"].tolist()
    metadata_list = df[["ID", "doc_id"]].to_dict(orient="records")
    ids = [f"doc_{i}" for i in range(1, len(content_list) + 1)]
    
    # contentとmetadataを表示
    print(content_list[:2])
    print(metadata_list[:2])

    # SentenceTransformersモデルでのEmbedding作成用関数
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-m3",
    )
    
    # VectorStoreの定義
    chroma_client = chromadb.PersistentClient("./testdb")

    collection = chroma_client.create_collection(
        name="my_collection",
        embedding_function = sentence_transformer_ef,
    )

    # バッチで登録する
    batches = create_batches(api=chroma_client, ids=ids, documents=content_list, metadatas=metadata_list)
    for batch in batches:
        print(f"Adding batch of size {len(batch[0])}")
        collection.add(ids=batch[0],
                       documents=batch[3],
                       #embeddings=batch[1],
                       metadatas=batch[2])

    #search_trial(collection)

    
def search_trial(collection):
    '''
    検索
    '''
    results = collection.query(
        query_texts=["テザリングしたい"],
        n_results=2
    )

    # 検索結果を表示
    #print(results)
    print(json.dumps(results, ensure_ascii=False))


def search (q_file, db):
    '''
    '''
    # SentenceTransformersモデルでのEmbedding作成用関数
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-m3",
    )

    # Persistent Clientでクライアント初期化
    chroma_client = chromadb.PersistentClient(db)

    # get_collectionで既存のコレクションにアクセス
    collection = chroma_client.get_collection(
        name="my_collection",
        embedding_function = sentence_transformer_ef,        
    )

    with open(q_file, 'r', encoding='utf-8') as fp:
        for line in fp:
            line = line.strip()
            eles = line.split('\t')
            correct_docid = eles[0]
            question = eles[1]
            #print(f'{correct_docid} --> {question}')

            # 検索
            results = collection.query(
                query_texts=[question],        
                n_results=10
            )
            print(json.dumps(results, ensure_ascii=False))

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True)    # Target Data_Dir
    parser.add_argument('-search', '--search', action='store_true')    # flag for search
    parser.add_argument('-db', '--db', required=False)    # Vector Store
    args = parser.parse_args()
    tgtfile = args.file
    flag_search = args.search
    file_db = args.db

    if flag_search:
        search (tgtfile, file_db)
    else:
        main(tgtfile)

