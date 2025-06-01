import pandas as pd
import numpy as np
from datasets import load_dataset
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import Normalizer


class chatBot :
    def __init__(self,dataset_loader):

        self.dataset = dataset_loader.load('Kamtera/Persian-conversational-dataset')
        text = [item['text'] for item in self.dataset['train']]

        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        embedding = model.encode(text,show_progress_bar=True)   


        normalize = Normalizer() 
        normalize = normalize.fit_transform(embedding)
        
        print(normalize.shape)
        