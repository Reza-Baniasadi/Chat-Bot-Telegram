from pymongo import MongoClient

class MongoDb:
    def __init__(self,uri="mongodb://localhost:27017/",db_name="chat_bot",collection_name="question"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]


    def insert_question(self,question:str,vector:list):
        data = {"question": question, "vector": vector}
        self.collection.insert_one(data)

        
    def get_all_question(self):
        return list(self.collection.find({}, {"_id": 0}))
        

    def findByQuestion(self,question_text):
        return self.collection.find_one({'question':question_text}, {"_id": 0})




