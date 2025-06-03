import pandas as pd
import numpy as np
from datasets import load_dataset
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import Normalizer


class ChatBot :
    def __init__(self):

        self.dataset = load_dataset('Kamtera/Persian-conversational-dataset',trust_remote_code=True)
        print('ddddddddddd',self.dataset['train'][0])
        text = [item['question'] for item in self.dataset['train']]



        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        embedding = model.encode(text,show_progress_bar=True)   


        normalize = Normalizer() 
        normalize = normalize.fit_transform(embedding)
        
        print("تعداد جملات:", len(text))
        print("شکل embedding ها:", embedding.shape)
        print("نمونه بردار embedding جمله اول (۱۰ مقدار اول):", embedding[0][:10])  
               
if __name__ == "__main__":
     chatbot = ChatBot()

 