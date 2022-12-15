import sys
import time
import csv
import math
import numpy as np
from pyspark import SparkContext
import json
from operator import add
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from node2vec import Node2Vec as n2v
import networkx as nx
from hashlib import md5
import pickle
import os


# environment setting
################ Uncomment if run on local
# os.environ["SPARK_HOME"] = "/Applications/spark-3.1.2-bin-hadoop3.2"
# os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.6"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/bin/python3.6"

def PartHash(string):
    seed = 131
    hash = 0
    for ch in string:
        hash = hash * seed + ord(ch)
    return hash & 0x7FFFFFFF

if __name__ == '__main__':
    # set the path for reading and outputting files
    folder_path = sys.argv[1]
    test_filepath = sys.argv[2]
    output_filepath = sys.argv[3]

    # Uncommon when run at local machine
    # folder_path = "datasets/"
    # test_filepath = "datasets/yelp_val.csv"
    # output_filepath = "../output_task2_2.csv"

    # connect the spark and set the environment
    sc = SparkContext('local[*]', 'graphembedding').getOrCreate()
    sc.setLogLevel("ERROR")

    # prepare datasets
    yelp_train = folder_path + "yelp_train.csv"
    ############## only uncomment if train on local (yelp_test is a random dataset created on business_train)
    # yelp_test = folder_path + "yelp_test.csv"
    user = folder_path + "user.json"
    business = folder_path + "business.json"


    start_time = time.time()
    # '''
    # read the dataset
    # train_dataset (uid_bid_stars)
    rdd1 = sc.textFile(yelp_train)
    head = rdd1.first()
    train_rdd = rdd1.filter(lambda x: x != head).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1], float(x[2])))
    # val_dataset
    rdd2 = sc.textFile(test_filepath)
    head2 = rdd2.first()
    val_rdd = rdd2.filter(lambda x: x != head2).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1]))
    # test_dataset
    ################### Only uncomment if train on local.
    # rdd3 = sc.textFile(yelp_test)
    # head3 = rdd3.first()
    # test_rdd = rdd3.filter(lambda x: x != head3).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1]))
    # business_dataset
    business_rdd = sc.textFile(business).map(lambda line: json.loads(line)).map(lambda x: (x['business_id'], x["categories"]))
    # user_dataset
    user_rdd = sc.textFile(user).map(lambda line: json.loads(line)).map(lambda x: (x["user_id"], x["friends"]))

    # for all the ids we currently have
    # business_ids
    business_list = list(set(train_rdd.map(lambda x: x[1]+"_bid").collect() + val_rdd.map(lambda x: x[1]+"_bid").collect()))
    ################### only uncomment if train on local.
    # business_list = list(set(train_rdd.map(lambda x: x[1] + "_bid").collect() + val_rdd.map(lambda x: x[1] + "_bid").collect() + test_rdd.map(lambda x: x[1] + "_bid").collect()))
    # user_ids
    user_list = list(set(train_rdd.map(lambda x: x[0] + "_uid").collect() + val_rdd.map(lambda x: x[0] + "_uid").collect()))
    ################### only uncomment if train on local.
    # user_list = list(set(train_rdd.map(lambda x: x[0]+"_uid").collect() + val_rdd.map(lambda x: x[0] + "_uid").collect() + test_rdd.map(lambda x: x[1] + "_uid").collect()))

    # deal with business and user dataset
    remaining_business = business_rdd.partitionBy(10, PartHash) \
        .map(lambda x: (x[0]+"_bid",x[1])) \
        .filter(lambda x: x[0] in business_list) \
        .map(lambda x: (x[0], x[1].replace("&"," ").replace("/"," ").replace("(","").replace(")","").replace("  ","").split(",") if x[1] is not None else ""))\
        .map(lambda x: (x[0], [h.strip().lower() for h in x[1]]))

    remaining_user = user_rdd.partitionBy(10, PartHash) \
        .map(lambda x: (x[0] + "_uid", [single.strip()+"_uid" for single in x[1].split(",")])) \
        .filter(lambda x: x[0] in user_list) \
        .map(lambda x: (x[0], [i for i in x[1] if i in user_list]))

    print("BEFORE JOIN PAIRS HAVE FINISHED!")

    # build business pairs and friend pairs
    remaining_business_list = remaining_business.collect()
    business_label_pair = set()
    for single_ele in remaining_business_list:
        b_id = single_ele[0]
        corr_labels = single_ele[1]
        if corr_labels != []:
            for label  in corr_labels:
                if label != "":
                    business_label_pair.add(tuple(sorted([b_id, md5(label.encode(encoding='UTF-8')).hexdigest()])))

    remaining_user_list = remaining_user.collect()
    user_friend_pair = set()
    for single_ele in remaining_user_list:
        u_id = single_ele[0]
        u_friends = single_ele[1]
        if u_friends != []:
            for friend in u_friends:
                if friend != "" and friend != "_uid":
                    user_friend_pair.add(tuple(sorted([u_id, friend])))
    new_final_list = list(business_label_pair) + list(user_friend_pair)

    # for all original pairs
    train_pair = train_rdd.map(lambda x: (x[0],x[1])).collect()
    val_pair = val_rdd.collect()
    ################### only uncomment if train on local.
    # test_pair = test_rdd.collect()
    # final_list = train_pair+val_pair+test_pair
    final_list = train_pair + val_pair
    print("FINISH BUILDING THE PAIRS.")

    # generating relationship
    with open('GraphEdgeInfo.txt', 'w+') as f:
        for pair_original in final_list:
            f.write(f'{pair_original[0]+"_uid"} {pair_original[1]+ "_bid"}\n')
        for pair_new in new_final_list:
            f.write(f'{pair_new[0]} {pair_new[1]}\n')
    print("FINISH WRITING THE PAIRS.", "TIME:", time.time()-start_time)
    
    # generate graph
    # start_time = time.time()
    graph = nx.read_edgelist('GraphEdgeInfo.txt', create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    print("FINISH BUILDING THE GRAPH", "TIME:", time.time()-start_time)
    # embeddings
    model = n2v(graph, dimensions=200, walk_length=25, num_walks=250, p=0.25, q=4, workers=16)
    # model = pickle.load(open("../competition/Node2Vec", "rb"))
    model2 = model.fit(window=15)#, min_count=1, batch_words=4)  # train model
    pickle.dump(model2,open("Node2Vec","wb"))
    print("FINISH TRAINING THE MODEL", "TIME:", time.time() - start_time)
    # '''
    model2 = pickle.load(open("Node2Vec","rb"))
    # print(list(model2.wv.get_vector("QfWFxmXqRGixztgaZN0gOA_bid")))
    # print(type(model2.wv.get_vector("QfWFxmXqRGixztgaZN0gOA_bid")))
    info_dict = {}
    with open("GraphEdgeInfo.txt","r") as f:
        for line in f.readlines():
            for single in line.split():
                if single in info_dict:
                    continue
                else:
                    info_dict[single] = list(model2.wv.get_vector(single))

    df_embeddings = pd.DataFrame.from_dict(info_dict, orient='index', columns=[f'E_{_}' for _ in range(200)]).reset_index().rename(columns={'index': 'id'})
    df_embeddings.to_csv('VectorizedFeatures.csv', index=False)


