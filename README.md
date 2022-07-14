# **Disaster Tweets Classifier**

## **Abstract**
在這份專案中，主要嘗試以BERT ( Bidirectional Encoder Representations from Transformers ) 進行NLP ( Natural Language Processing ) 任務。資料集為kaggle上的Disaster Tweets ( https://www.kaggle.com/competitions/nlp-getting-started/overview ) 。在Disaster Tweets的資料集內蒐集了在推特上各式各樣的推文，而任務的目標是期望能使模型判斷該推文是否發布了一項災難事件 ( Disaster ) ，是=1；否=0。

## **Methodology**

在方法上使用BERT作為嵌入層 ( Embedding Layer ) 來取得輸入序列的向量表示 ( Vector Representation)，並取每個序列輸出向量在 \[CLS\] 向量表示再進入線性層當中進行分類預測。