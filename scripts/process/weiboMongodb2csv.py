"""
Script to extract data from MongoDB and save it as a compressed JSON file.
"""

import pandas as pd
from pymongo import MongoClient

from environ.constants import DATA_PATH

# Connect to MongoDB of local mongodb
CLIENT = MongoClient('mongodb://localhost:27017/')
db = CLIENT['weibo']
collection = db['weibo']

corpus = []

# Iterate over each document in the collection
for document in collection.find():
    corpus.append(document)

# Connect to MongoDB of remote mongodb
CLIENT = MongoClient('mongodb://localhost:27018/')
db = CLIENT['weibo']
collection = db['weibo']

# Iterate over each document in the collection
for document in collection.find():
    corpus.append(document)

# Convert the corpus to a DataFrame
corpus = pd.DataFrame(corpus)[["text"]]
corpus.to_csv(DATA_PATH / "weibo.csv", index=False)
