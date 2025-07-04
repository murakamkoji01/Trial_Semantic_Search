# Trial_Semantic_Search
Semantic Search Sample (LangChain+bge-m3+Chroma, designed for RMAA)

## 概要
Semantic Searchの実装を試した
* LangChain + bge-m3 + Chroma
  * LangChain : Framework
  * bge-m3 : Embedding Model
  * Chroma : Vector Store


## ファイル構成（主要なスクリプト）
* requirements.txt （後ほど）
  * pipでインストールするライブラリを記述
 
* src/get_data.py
  * JSONLのデータから必要な情報を抽出してファイル化
  * RecursiveCharacterTextSplitter でファイルをチャンク化
  * `$ python3 src/get_data.py -f hogehoge.jsonl`

* src/search.py
  * bge-m3でベクトルした文書情報をChromaのIndexingで格納
    * データからIndexを作成する
    * `$ python3 src/search.py -f data.jsonl`

  * Vector Indexを使って検索
    * Indexにクエリを与えて検索する
    * `$ python3 src/search.py -search -db mydb -f queries.txt > result.txt`

* src/mk_data.py
  * bge-m3モデルのFine-tuneのためのデータ作成用
    * Reference: [https://bge-model.com/index.html](https://bge-model.com/index.html)





