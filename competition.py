"""
Method Description:
This is a hybrid system considering entities relationships, business features and user features.

- Step1: With given business-business relationships, business-user relationships, and business-categories relationships,
we build a graph that makes connection among them. Then, graph embedding technique is used to construct 200 dimension numeric
vectors that can represent the relationships between them.
(File name: 'GraphEmbedding.py'; Output: 'VectorizedFeatures.csv')
- Step2: Categories of restaurants shouldn't be viewed as separated features. Therefore, I add them back to the business
vectors as side information to enhance the expression of individual business. (NOTE: Because the memory limit of Vocareum,
I incorporate this step into training phase files.)
- Step3: Two machine learning models are trained based on the embedding factors, business features, and user features.
(File name: 'competition_XGBOOST.py'; Output: 'XGBoostRegWithEmbedV4Long' - pickle file)
(File name: 'competition_CatBOOST.py'; Output: 'CatGBMRegWithEmbedV4Long' - pickle file)
- Step4: Outputs of the models are combined. We replace prediction scores that >5 as 5 and prediction scores that <1 as 1
before combination and after combination. Then we get our final results!

(Special Notes: (1) Including variations of ratings for users and business will dramatically damage the final result.
(2) If possible, use LightGBM will dramatically improve the performance and efficiency of model.
(3) Without memory limit of Vocareum, using 300 dimension vectors of embedding process will be the best choice.
(4) More data will allow the model to learn more patterns, valid for each step!)

Execution Time:
300s - Only main file
(Because of the complexity of deep walk technique, cross validation, and grid search, the supplementary documents may cost
about 6 hours to build the model. - Faster if you have GPU or use XGBOOST V1.7.)
"""
import sys
import time
import csv
import math
import numpy as np
from pyspark import SparkContext
import json
from operator import add
import pandas as pd
from hashlib import md5
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import os
import pickle

# environment setting

# os.environ["SPARK_HOME"] = "/Applications/spark-3.1.2-bin-hadoop3.2"
# os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.6"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/bin/python3.6"

def dealWithNaN(x,pos,num):
    if x[pos] is None:
        return num
    else:
        return x[pos]

p_similar = {}
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
    # output_filepath = "output_task2_3_add.csv"

    # connect the spark and set the environment
    sc = SparkContext('local[*]', 'COMPETITION').getOrCreate()
    sc.setLogLevel("ERROR")

    # prepare datasets
    yelp_train = folder_path + "yelp_train.csv"
    # yelp_valid = folder_path + "yelp_val.csv"
    user = folder_path + "user.json"
    business = folder_path + "business.json"
    photo = folder_path + "photo.json"
    # checkin = folder_path + "checkin.json"
    review_train = folder_path + "review_train.json"
    # tip = folder_path + "tip.json"
    vec_features = "VectorizedFeatures.csv"

    start_time = time.time()
    # import dataset!
    # train_dataset
    rdd1 = sc.textFile(yelp_train)
    head = rdd1.first()
    uid_bid_star = rdd1.filter(lambda x: x != head).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1], float(x[2])))
    bid = uid_bid_star.map(lambda x: (x[1], 1)).distinct()
    uid = uid_bid_star.map(lambda x: (x[0], 1)).distinct()
    # test_dataset
    rdd2 = sc.textFile(test_filepath)
    head2 = rdd2.first()
    uid_bid_need = rdd2.filter(lambda x: x != head2).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1]))
    # uid_bid_need = rdd2.filter(lambda x: x != head2).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1], float(x[2])))
    bid_test = uid_bid_need.map(lambda x: (x[1], 1)).distinct()
    uid_test = uid_bid_need.map(lambda x: (x[0], 1)).distinct()
    # business_dataset
    rdd3 = sc.textFile(business).map(lambda line: json.loads(line)).map(
        lambda x: (x['business_id'], x['review_count'], x['stars'], x["city"], x["is_open"]))
    # photo dataset
    rdd4 = sc.textFile(photo).map(lambda line: json.loads(line)).map(
        lambda x: (x['business_id'], x['photo_id'], x['label']))
    # review_train dataset
    rdd5 = sc.textFile(review_train).map(lambda line: json.loads(line)).map(
        lambda x: (x['user_id'], x['stars'], x['useful'], x['funny'], x['cool'], x["business_id"]))
    # user dataset
    rdd6 = sc.textFile(user).map(lambda line: json.loads(line)).map(lambda x: (
        x["user_id"], x["review_count"], x["average_stars"], x['fans'], x["friends"], x["useful"], x["funny"],
        x["cool"],
        x["compliment_hot"], x["compliment_more"], x["compliment_profile"], x["compliment_cute"], x["compliment_list"],
        x["compliment_note"], x["compliment_plain"], x["compliment_cool"],
        x["compliment_funny"], x["compliment_writer"], x["compliment_photos"], x["yelping_since"]))
    # vec features import
    df_features = pd.read_csv(vec_features)
    # df_collFilter = pd.read_csv(addColFileterData)

    # build the vars needed
    # from business perspective
    # types of photos business have
    restaurant_photo = rdd4.filter(lambda x: x[2] in ["inside", "outside", "food", "drink"]).map(
        lambda x: (x[0], 1)).reduceByKey(add)
    # number of reviews business have
    # use the whole dataset
    bid_num_review = rdd3.map(lambda x: (x[0], x[1])).reduceByKey(add)
    # only use the review_train dataset
    # bid_num_review = rdd5.map(lambda x: (x[5],1)).reduceByKey(add)
    # average stars business have
    # use whole dataset
    bid_avg_stars = rdd3.map(lambda x: (x[0], x[2]))
    # only use review_train dataset
    # bid_avg_stars = rdd5.map(lambda x: (x[5], x[1])).groupByKey().mapValues(list).mapValues(lambda x: np.mean(x))
    # variance of stars for a business
    bid_var_stars = rdd5.map(lambda x: (x[5], x[1])).groupByKey().mapValues(list).mapValues(lambda x: np.var(x))

    # print(bid_avg_stars.take(3))
    # from user perspectives
    # number of creditability
    # num_fans_user = rdd5.map(lambda x: (x[0], x[2]+x[3]+x[4])).reduceByKey(add)
    num_fans_user = rdd6.map(lambda x: (x[0], x[3]))
    # number of reviews user write
    # uid_num_review = rdd5.map(lambda x: (x[0], 1)).reduceByKey(add)
    uid_num_review = rdd6.map(lambda x: (x[0], x[1]))
    # average stars of users
    # uid_avg_stars = rdd5.map(lambda x: (x[0],x[1])).reduceByKey(add).leftOuterJoin(uid_num_review).map(lambda x: (x[0],x[1][0]/x[1][1]))
    uid_avg_stars = rdd6.map(lambda x: (x[0], x[2]))
    # variance of stars for a user
    uid_var_stars = rdd5.map(lambda x: (x[0], x[1])).groupByKey().mapValues(list).mapValues(lambda x: np.var(x))
    # number of friends a user have
    uid_num_friends = rdd6.map(lambda x: (x[0], len(x[4].split(","))))
    # number of interations for a user
    uid_num_iteractions = rdd6.map(lambda x: (x[0], (x[5], x[6], x[7])))
    # number of compliments recieved
    uid_num_compliment = rdd6.map(
        lambda x: (x[0], (x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18])))
    # yelp since
    uid_yelp_since = rdd6.map(lambda x: (x[0], (2022 - int(x[19][:4]))))
    # retaurant is_open? & state bid
    bid_open_state = rdd3.map(lambda x: (x[0], (x[3], x[4])))

    clf = pickle.load(open("XGBoostRegWithEmbedV4Long","rb"))
    clf2 = pickle.load(open("CatGBMRegWithEmbedV4Long","rb"))
    bid_final_test = bid_test.leftOuterJoin(restaurant_photo).map(lambda x: (x[0], x[1][1])).map(
        lambda x: (x[0], dealWithNaN(x, 1, 0))) \
        .leftOuterJoin(bid_num_review).map(lambda x: (x[0], x[1][0], x[1][1])).map(
        lambda x: (x[0], (x[1], dealWithNaN(x, 2, 0)))) \
        .leftOuterJoin(bid_avg_stars).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], dealWithNaN(x, 3, 3)))) \
        .leftOuterJoin(bid_var_stars).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], x[3], dealWithNaN(x, 4, 0)))) \
        .leftOuterJoin(bid_open_state)\
        .map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], dealWithNaN(x[1],1,("NULL",1))))\
        .map(lambda x: (x[0], (x[1], x[2], x[3], x[4], x[5][0], x[5][1])))
    # for uid (treat all null user as 3)
    uid_final_test = uid_test.leftOuterJoin(num_fans_user).map(lambda x: (x[0], x[1][1])).map(
        lambda x: (x[0], dealWithNaN(x, 1, 0))) \
        .leftOuterJoin(uid_num_review).map(lambda x: (x[0], x[1][0], x[1][1])).map(
        lambda x: (x[0], (x[1], dealWithNaN(x, 2, 0)))) \
        .leftOuterJoin(uid_avg_stars).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], dealWithNaN(x, 3, 3)))) \
        .leftOuterJoin(uid_var_stars).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], x[3], dealWithNaN(x, 4, 0)))) \
        .leftOuterJoin(uid_num_friends).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], x[3], x[4], dealWithNaN(x, 5, 1)))) \
        .leftOuterJoin(uid_num_iteractions) \
        .map(lambda x: (
    x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], dealWithNaN(x[1], 1, (0,0,0)))) \
        .map(lambda x: (
    x[0], (x[1], x[2], x[3], x[4], x[5], x[6][0], x[6][1], x[6][2]))) \
        .leftOuterJoin(uid_num_compliment) \
        .map(lambda x: (
    x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5], x[1][0][6], x[1][0][7],
           dealWithNaN(x[1], 1, (0,0,0,0,0,0,0,0,0,0,0)))) \
        .map(lambda x: (
        x[0], (x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8],
               x[9][0], x[9][1], x[9][2], x[9][3], x[9][4], x[9][5], x[9][6], x[9][7], x[9][8], x[9][9], x[9][10]))) \
        .leftOuterJoin(uid_yelp_since) \
        .map(lambda x: (
    x[0], (x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5], x[1][0][6], x[1][0][7],
           x[1][0][8], x[1][0][9], x[1][0][10], x[1][0][11], x[1][0][12], x[1][0][13], x[1][0][14], x[1][0][15],
           x[1][0][16], x[1][0][17], x[1][0][18], dealWithNaN(x[1],1,4))))

    # final format: uid, bid, num_photo, bid_num_review, bid_avg_stars, uid_num_fans, uid_num_reviews, uid_avg_stars, actual_stars
    uid_bid_star_final_test = uid_bid_need.map(lambda x: (x[1], (x[0]))).leftOuterJoin(bid_final_test) \
        .map(lambda x: (x[1][0], (x[0], x[1][1][0], x[1][1][1], x[1][1][2], x[1][1][3], x[1][1][4], x[1][1][5]))) \
        .leftOuterJoin(uid_final_test) \
        .map(lambda x: (
    x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5], x[1][0][6], x[1][1][0], x[1][1][1],
    x[1][1][2], x[1][1][3],
    x[1][1][4], x[1][1][5], x[1][1][6], x[1][1][7], x[1][1][8], x[1][1][9], x[1][1][10], x[1][1][11], x[1][1][12],
    x[1][1][13], x[1][1][14],
    x[1][1][15], x[1][1][16], x[1][1][17], x[1][1][18], x[1][1][19])).collect()

    # print(uid_bid_star_final_test)
    # exit()
    # prepare for the machine learning train data
    df_test_features_add = pd.DataFrame({"uid": [p[0] + "_uid" for p in uid_bid_star_final_test],
                                         "bid": [p[1] + "_bid" for p in uid_bid_star_final_test],
                                         "num_photo": [p[2] for p in uid_bid_star_final_test],
                                         "bid_num_review": [p[3] for p in uid_bid_star_final_test],
                                         "bid_avg_stars": [p[4] for p in uid_bid_star_final_test],
                                         "bid_open": [p[6] for p in uid_bid_star_final_test],
                                         "bid_city": [p[7] for p in uid_bid_star_final_test],
                                         "uid_num_fans": [p[8] for p in uid_bid_star_final_test],
                                         "uid_num_reviews": [p[9] for p in uid_bid_star_final_test],
                                         "uid_avg_stars": [p[10] for p in uid_bid_star_final_test],
                                         "uid_num_friends": [p[12] for p in uid_bid_star_final_test],
                                         "uid_interactions1": [p[13] for p in uid_bid_star_final_test],
                                         "uid_interactions2": [p[14] for p in uid_bid_star_final_test],
                                         "uid_interactions3": [p[15] for p in uid_bid_star_final_test],
                                         "uid_compliment1": [p[16] for p in uid_bid_star_final_test],
                                         "uid_compliment2": [p[17] for p in uid_bid_star_final_test],
                                         "uid_compliment3": [p[18] for p in uid_bid_star_final_test],
                                         "uid_compliment4": [p[19] for p in uid_bid_star_final_test],
                                         "uid_compliment5": [p[20] for p in uid_bid_star_final_test],
                                         "uid_compliment6": [p[21] for p in uid_bid_star_final_test],
                                         "uid_compliment7": [p[22] for p in uid_bid_star_final_test],
                                         "uid_compliment8": [p[23] for p in uid_bid_star_final_test],
                                         "uid_compliment9": [p[24] for p in uid_bid_star_final_test],
                                         "uid_compliment10": [p[25] for p in uid_bid_star_final_test],
                                         "uid_compliment11": [p[26] for p in uid_bid_star_final_test],
                                         "uid_yelp_since": [p[27] for p in uid_bid_star_final_test]})
    df_test_features_add = pd.get_dummies(df_test_features_add, columns=["bid_city"]).drop(["bid_open"], axis=1)

    # df_xg_test = pd.read_csv("xg_final.csv")
    business_rdd_cf = sc.textFile(business).map(lambda line: json.loads(line)).map(
        lambda x: (x['business_id'], x['categories']))

    remaining_business = business_rdd_cf.partitionBy(10, PartHash) \
        .map(lambda x: (x[0] + "_bid", x[1])) \
        .map(lambda x: (x[0], x[1].replace("&", " ").replace("/", " ").replace("(", "").replace(")", "").replace("  ",
                                                                                                                 "").split(
        ",") if x[1] is not None else "")) \
        .map(lambda x: (x[0], [h.strip().lower() for h in x[1]])).flatMap(
        lambda x: map(lambda type: (x[0], type), x[1])) \
        .map(lambda x: (md5(x[1].encode(encoding='UTF-8')).hexdigest(), x[0])).collect()

    # transfer to data frame
    type_res = [i[0] for i in remaining_business]
    id_cf = [i[1] for i in remaining_business]
    df_conn1 = pd.DataFrame({"type": type_res, "bid_cat": id_cf})

    add_info = list(set(list(df_conn1["bid_cat"])))
    df_conn_add = pd.DataFrame({"type": add_info, "bid_cat": add_info})
    df_conn = pd.concat([df_conn1, df_conn_add])

    # merge two datasets
    catfeatures = df_conn.merge(df_features, left_on="type", right_on="id", how="left").drop(["type", "id"],
                                                                                             axis=1).fillna(0). \
        groupby("bid_cat").mean()
    catfeatures.reset_index(inplace=True)
    df_test_features = df_test_features_add \
        .merge(df_features, left_on="uid", right_on="id", how="left") \
        .drop(['uid', 'id'], axis=1) \
        .merge(catfeatures, left_on="bid", right_on="bid_cat", how="left") \
        .drop(["bid", "bid_cat"], axis=1).fillna(0)
    y_pred = clf.predict(df_test_features)
    y_pred2 = clf2.predict(df_test_features)
    '''
    df_final = pd.DataFrame(
        {"user_id": [p[0] for p in uid_bid_star_final_test], "business_id": [p[1] for p in uid_bid_star_final_test], "num_photo": [p[2] for p in uid_bid_star_final_test], "bid_num_review": [p[3] for p in uid_bid_star_final_test],
         "bid_avg_stars": [p[4] for p in uid_bid_star_final_test], "bid_var": [p[5] for p in uid_bid_star_final_test],"uid_num_fans": [p[6] for p in uid_bid_star_final_test],
         "uid_num_reviews": [p[7] for p in uid_bid_star_final_test], "uid_avg_stars": [p[8] for p in uid_bid_star_final_test],"uid_var":[p[9] for p in uid_bid_star_final_test],
         "uid_num_friends":[p[10] for p in uid_bid_star_final_test], "uid_num_interactions": [p[11] for p in uid_bid_star_final_test],"uid_num_compliment": [p[12] for p in uid_bid_star_final_test], "uid_yelp_since":[p[13] for p in uid_bid_star_final_test],
         "prediction": y_pred})
    '''
    df_final = pd.DataFrame({"user_id":[p[0] for p in uid_bid_star_final_test],"business_id":[p[1] for p in uid_bid_star_final_test],"prediction_x": y_pred, "prediction_cat": y_pred2})
    # print("testing_RMSE:",mean_squared_error(y_true=df_test_target['score'], y_pred=clf.predict(df_test_features_scaled),squared=False))



    df = df_final
    # df.to_csv("xg&cat.csv")

    info = []
    for i in range(len(df["prediction_x"])):
        if df.iloc[i,2] >=5 and df.iloc[i,3]>=5:
            info.append(5)
        elif df.iloc[i,2] <=1 and df.iloc[i,3]<=1:
            info.append(1)
        else:
            a = 0.4*df.iloc[i,2] + 0.6*df.iloc[i,3]
            if a<1:
                info.append(1)
            elif a>5:
                info.append(5)
            else:
                info.append(a)

        '''
        if df["prediction_y"][i] == -1:
            info.append(0.5*df["prediction_x"][i]+0.5*df["prediction_cat"][i])
        else:
            # info.append(df["prediction_x"][i] * 0.4 + df["prediction_cat"][i] * 0.5)
            info.append(df["prediction_x"][i] * 0.5 + df["prediction_cat"][i] * 0.5)
            # info.append(df["prediction_x"][i] * 0.5 + df["prediction_cat"][i] * 0.5)
            # info.append(df["prediction_y"][i] * 0.1 + df["prediction_x"][i] * 0.9)
        '''
    df["prediction"] = info
    df_final = df[["user_id", "business_id", "prediction"]]
    df_final.to_csv(output_filepath,index=False)
    print("Duration:", time.time()-start_time)

    # df2 = df_final
    # df3 = pd.read_csv(test_filepath)

    # df = df3.merge(df2, on=["user_id", "business_id"])
    # df["diff"] = abs(df["stars"] - df["prediction"])
    # df["diff_2"] = [i ** 2 for i in df["diff"]]
    # print((sum(list(df["diff_2"])) / len(list(df["diff_2"]))) ** 0.5)


# 0.98401752 Nov 1 0.9*XG-BOOST AND 0.1*ITEM_BASED
# 0.98350688 Nov 1 0.8*XG-BOOST AND 0.1*ITEM_BASED AND 0.1*CAT-BOOST
# 0.98346499 Nov 1 0.8*XG-BOOST AND 0.1*ITEM_BASED AND 0.1*CAT-BOOST; 0.1*rate+0.6*bid+0.3*uid
# 0.9831146897 NOV 1 use xgboost and cat boost with new parameters
# 0.983071806317673 NOV 5 use 5,8 bound, 3.5avg in coll filtering; new xg boost hyper paras
# 0.9738836654111553 NOV 30 with xg 0.4 and cat 0.6
# 0.97389033983645 NOV 30 xg 0.5 and cat 0.5
# 0.973582478383629 Dec 1 xg 0.5 and cat 0.5
# 0.9370940544852785 for validation
# 0.958297976188725 for simulate test 1
# 0.9923226137489639 for simulate test 2
# 0.9664899402246429 for simulate test 3

