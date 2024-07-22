"""
@ Author: Hou Yuxin
@ Date: July 16th, 2024
@ Target: Drag several hot debated topics from bangdan data

Bangdan File Structure
```
|-bangdan_data
|---year
|-----weibo_bangdan.yyyy-mm-dd
```

Bangdan Data Fromat
```
61543(integer)\tjson_data_wrap1\n
```

json_data_wrap1
```
{
    "id": xxxxx,
    "crawler_time": xxxxx,
    "crawler_time_stamp": xxxx,
    "type": 1, # 1-real time, 2-hottest
    "bangdan": json_data_wrap2,
    "date": xxxx
}
```

json_data_wrap2 为节约空间，仅显示相关内容
```
{
    "cards": [], #n个卡片
    "cardlistInfo": {},
    xxx
}
```

cards 仅考虑card_type == 11的内容，是实际展示的榜单
```
{
    "card_type": 11,
    "title": "xxx",
    "show_type": xx,
    "card_group": [
        {
            "card_type": 4,
            "pic": "",
            "desc": "xxx" #需要的就是这个desc
        }
    ],
    "openurl"
}
```

当前方案：
考虑以下几个措施 - 
- 分年做分析？分月份做分析
- 分别分析最热榜、实时榜
- 每一个榜单仅取前10
- 去重和不去重两个版本

"""

import os
import glob
import json
import random
import numpy as np
from datetime import datetime

from configs import *
from utils.utils import extract_7z_files

from typing import Dict, List, Optional, Tuple

import jieba

import torch

from transformers import BertTokenizer, BertModel

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

import gensim
from gensim.models.coherencemodel import CoherenceModel


def get_bangdan_files_dir(year):
    return f"{DATA_SOURCE_DIR}/{year}/bangdan/"


def get_bangdan_unzipped_files_dir(year):
    return f"bangdan_data/{year}/"


def unzip_all_bangdan_files():
    """
    将原始微博数据解压缩到当前目录的bangdan_data文件夹
    """
    for year in [2023]: #ANALYSIS_YEARS:
        bangdan_files_dir = get_bangdan_files_dir(year)
        unzipped_dir = get_bangdan_unzipped_files_dir(year)
        extract_7z_files(source_folder=bangdan_files_dir, target_folder=unzipped_dir)


class BangdanAnalyzer(object):

    def __init__(
        self,
        year: int,
        by_month: bool = False, # 是否按月份分别分析，默认为否
        bangdan_type: List[str] = None, # 1 - realtime, 2 - hottest
        top10: bool = True, # 榜单仅取前10 TODO 试试取前20
        skip_repetition: bool = True, # TODO 是否去重，默认为是
        
    ):
        if bangdan_type is None:
            bangdan_type = ["1", "2"]
        self.year = year
        self.data_dir = get_bangdan_unzipped_files_dir(year)
        self.by_month = by_month
        self.bangdan_type = bangdan_type
        self.top10 = top10
        self.skip_repetition = skip_repetition
    
    
    def get_file_list(self, month: int = None):
        if self.by_month and month is None:
            raise ValueError("please specify the month")
        if self.by_month:
            pattern = os.path.join(self.data_dir, f"weibo_bangdan.{self.year}-{str(month).zfill(2)}-??")
        else:
            pattern = os.path.join(self.data_dir, f"weibo_bangdan.{self.year}-??-??")
        
        matching_files = glob.glob(pattern)
        return matching_files
    

    def get_bangdan_text_from_file_list(self, file_list: List[str]):
        # bangdan_text_list = []
        bangdan_text_set = set()
        for file_name in file_list:
            with open(file_name, "r", errors="replace") as rfile:
                for line in rfile.readlines():
                    line = line.strip()
                    line_data = line.split("\t")
                    if len(line_data) < 2:
                        print("line data cannot be splitted")
                        continue
                    try:
                        data = json.loads(line_data[1])
                    except json.JSONDecodeError as e:
                        print(f"JSONDecodeError: {e}")
                        # 打印出错误位置
                        print(f"Error at line {e.lineno}, column {e.colno}")
                        # 打印出错误字符位置
                        print(f"Error at character {e.pos}, {line_data[1][int(e.pos)-20: int(e.pos)+20]}")
                        continue
                    if data["type"] not in self.bangdan_type:
                        # print(f"wrong data type: {data['type']}")
                        # 排除不允许的榜单类型
                        continue
                    data = json.loads(data["bangdan"])
                    if type(data) is not dict:
                        print(f"bad data type")
                        print(data)
                        continue
                    if "cards" not in data.keys() or data["cards"] is None:
                        print(f"bad data type in file {file_name}")
                        continue
                    for card in data["cards"]:
                        if str(card["card_type"]) != "11":
                            continue
                        card_group = card["card_group"]
                        if self.top10:
                            card_group = card_group[0:20]
                        for s_card in card_group:
                            if str(s_card["card_type"]) != "4":
                                continue
                            if "desc" in s_card.keys():
                                bangdan_text_set.add(s_card["desc"])
                            # elif "title" in s_card.keys():
                            #     bangdan_text_set.add(s_card["title"])
                            else:
                                print(f"desc not in keys! file_name {file_name}, data: {s_card}")
        return list(bangdan_text_set)
    

    def compute_coherence_values(self, X, dictionary, corpus, texts, limit, start=3, step=3):
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
            model.fit(X)
            model_list.append(model)

            topics = []
            for topic_idx, topic in enumerate(model.components_):
                topics.append([dictionary[i] for i in topic.argsort()[:-10 - 1:-1]])
            coherencemodel = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
        return model_list, coherence_values   
    

    def output_model_results(self, X, model, feature_names, n_top_words, texts, max_coherence_value, output_file_name, min_texts_per_topic=10,):
        doc_topic_dist = model.transform(X)
        topic_doc_count = (doc_topic_dist.argmax(axis=1)[:, None] == range(model.n_components)).sum(axis=0)
        
        now = datetime.now()
        output_text = f"Log on {now}: \n"
        for topic_idx, topic in enumerate(model.components_):
            if topic_doc_count[topic_idx] < min_texts_per_topic:
                continue
            output_text += f"Topic #{topic_idx} ({topic_doc_count[topic_idx]} texts):\n"
            output_text += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            output_text += "\n"
            
            # 获取属于当前主题的文本索引
            topic_text_indices = [i for i, topic_dist in enumerate(doc_topic_dist) if topic_dist.argmax() == topic_idx]
            # 随机抽取20条文本（如果不足20条则全部输出）
            sampled_text_indices = random.sample(topic_text_indices, min(20, len(topic_text_indices)))
            sampled_texts = [texts[i] for i in sampled_text_indices]
            
            output_text += "Sampled texts:\n"
            for text in sampled_texts:
                output_text += text + "\n"
            output_text += "\n\n\n"
        
        output_text += f"Perplexity: {model.perplexity(X)}\n"
        output_text += f"Log Likelihood: {model.score(X)}\n"
        output_text += f"Coherence Values: {max_coherence_value}"

        output_text += "\n\n\n\n"
        
        with open(output_file_name, "a", encoding="utf8") as wfile:
            wfile.write(output_text)


    def lda_analysis(self, text_list, output_file_name: str, n_top_words: int=10,):
        def tokenize(text):
            return ' '.join(jieba.cut(text))
        
        tokenized_texts = [tokenize(text) for text in text_list]

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(tokenized_texts)

        texts_tokenized = [text.split() for text in tokenized_texts]
        dictionary = gensim.corpora.Dictionary(texts_tokenized)
        corpus = [dictionary.doc2bow(text) for text in texts_tokenized]

        start, limit, step = 5, 15, 1
        model_list, coherence_values = self.compute_coherence_values(X=X, dictionary=dictionary, corpus=corpus, texts=texts_tokenized, start=start, limit=limit, step=step)

        best_model_index = coherence_values.index(max(coherence_values))
        best_model = model_list[best_model_index]
        tf_feature_names = vectorizer.get_feature_names_out()

        max_coherence_value = max(coherence_values)

        self.output_model_results(X, best_model, tf_feature_names, n_top_words, text_list, max_coherence_value, output_file_name=output_file_name, min_texts_per_topic=10)


    def load_pretrained_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.model = BertModel.from_pretrained("bert-base-chinese")
        self.model.eval()
        self.model.to(self.device)
    
    def get_bert_embedding(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    
    def cluster(self, text_list, output_file_name: str, num_clusters: int = 10):
        text_embeddings = np.array([
            self.get_bert_embedding(text) for text in text_list
        ])
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(text_embeddings)

        distances = cdist(text_embeddings, kmeans.cluster_centers_, 'euclidean')

        with open(output_file_name, "a", encoding="utf8") as wfile:
            for label in range(num_clusters):
                wfile.write(f"Label #{label}: \n")
                label_indices = [i for i in range(len(text_list)) if labels[i] == label]
                label_texts = [text_list[i] for i in label_indices]
                label_distances = [distances[i, label] for i in label_indices]
                # 根据距离进行排序
                sorted_indices = np.argsort(label_distances)
                sorted_texts = [label_texts[i] for i in sorted_indices]
                sorted_distances = [label_distances[i] for i in sorted_indices]
                # 输出前20个样本
                wfile.write(f'{len(sorted_texts)} samples of the texts belonged to this label\n')
                for text, distance in zip(sorted_texts[:20], sorted_distances[:20]):
                    wfile.write(f'{text} (distance: {distance:.4f})\n')
                wfile.write('\n\n')

        print(f'Cluster results saved to {output_file_name}')


    def analyze(self, analysis_mode: str = "kmeans"):
        # kmeans or lda
        if self.by_month is False:
            file_list = self.get_file_list()
            text_list = self.get_bangdan_text_from_file_list(file_list)
            self.lda_analysis(text_list, f"logs/{self.year}.out")
        else:
            for month_id in range(12):
                month = month_id + 1
                # if self.year == 2022 and month < 12:
                #     continue
                file_list = self.get_file_list(month=month)
                if len(file_list) == 0:
                    print(f"No file list: Year-{self.year}, Month-{month}")
                    continue
                text_list = self.get_bangdan_text_from_file_list(file_list)
                if analysis_mode == "lda":
                    self.lda_analysis(text_list, f"logs/{self.year}-{str(month).zfill(2)}.out")
                else:
                    self.cluster(text_list, f"logs/kmeans-{self.year}-{str(month).zfill(2)}.out")
                print(f"finished month {month}\n")


if __name__ == "__main__":
    # unzip_all_bangdan_files()
    for year in ANALYSIS_YEARS:
        # if year in [2020, 2021]:
        #     continue
        print(f"\n\nprocessing year-{year}")
        for by_month in [True]: #, False]:
            analyzer = BangdanAnalyzer(
                year=year,
                by_month=by_month,
            )
            analyzer.analyze()
