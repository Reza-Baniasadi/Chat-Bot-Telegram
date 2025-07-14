from pymongo import MongoClient

class MongoDb:
    def __init__(self,uri="mongodb://localhost:27017/",db_name="chat_bot",collection_name="question"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]




