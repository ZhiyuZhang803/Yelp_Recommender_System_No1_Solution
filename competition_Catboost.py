import sys
import time
from hashlib import md5
import csv
import math
import numpy as np
from pyspark import SparkContext
import json
from operator import add
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import os
from catboost import CatBoostRegressor
import pickle

# environment setting
########################### Uncomment only train on local.
# os.environ["SPARK_HOME"] = "/Applications/spark-3.1.2-bin-hadoop3.2"
# os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3.6"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/local/bin/python3.6"

def dealWithNaN(x,pos,num):
    if x[pos] is None:
        return num
    else:
        return x[pos]

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
    # output_filepath = "output_forCatBoost.csv"

    # connect the spark and set the environment
    sc = SparkContext('local[*]', 'task2_2').getOrCreate()
    sc.setLogLevel("ERROR")

    # prepare datasets
    yelp_train = folder_path + "yelp_train.csv"
    # yelp_valid = folder_path + "yelp_val.csv"
    user = folder_path + "user.json"
    business = folder_path + "business.json"
    photo = folder_path + "photo.json"
    # checkin = folder_path + "checkin.json"
    review_train = folder_path + "review_train.json"
    vec_features = "VectorizedFeatures.csv"
    # tip = folder_path + "tip.json"
    # addColFileterData = "BaseLine.csv"

    start_time = time.time()
    # import dataset!
    # train_dataset
    rdd1 = sc.textFile(yelp_train)
    head = rdd1.first()
    uid_bid_star = rdd1.filter(lambda x: x != head).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1], float(x[2])))
    # rdd_val = sc.textFile(yelp_valid)
    # head_val = rdd_val.first()
    # uid_bid_star_val = rdd_val.filter(lambda x: x != head_val).map(lambda x: x.split(",")).map(lambda x: (x[0], x[1], float(x[2])))
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
    x["user_id"], x["review_count"], x["average_stars"], x['fans'], x["friends"], x["useful"], x["funny"], x["cool"],
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
        lambda x: (x[0], (x[8] , x[9] , x[10] , x[11] , x[12] , x[13] , x[14] , x[15] , x[16] , x[17] , x[18])))
    # yelp since
    uid_yelp_since = rdd6.map(lambda x: (x[0], (2022 - int(x[19][:4]))))
    # retaurant is_open? & state bid
    bid_open_state = rdd3.map(lambda x: (x[0], (x[3], x[4])))
    # print(uid_avg_stars.take(3))
    # print(rdd3.collect())

    # join the data to create the train dataset
    # for bid (treat all null restaurant as 3)
    bid_final_train = bid.leftOuterJoin(restaurant_photo).map(lambda x: (x[0], x[1][1])).map(
        lambda x: (x[0], dealWithNaN(x, 1, 0))) \
        .leftOuterJoin(bid_num_review).map(lambda x: (x[0], x[1][0], x[1][1])).map(
        lambda x: (x[0], (x[1], dealWithNaN(x, 2, 0)))) \
        .leftOuterJoin(bid_avg_stars).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], dealWithNaN(x, 3, 3)))) \
        .leftOuterJoin(bid_var_stars).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], x[3], dealWithNaN(x, 4, 0)))) \
        .leftOuterJoin(bid_open_state).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][1][0], x[1][1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], x[3], x[4], dealWithNaN(x, 5, "NULL"), dealWithNaN(x, 6, 1))))

    # print(bid_final_train.collect())
    # exit()
    # for uid (treat all null user as 3)
    uid_final_train = uid.leftOuterJoin(num_fans_user).map(lambda x: (x[0], x[1][1])).map(
        lambda x: (x[0], dealWithNaN(x, 1, 0))) \
        .leftOuterJoin(uid_num_review).map(lambda x: (x[0], x[1][0], x[1][1])).map(lambda x: (x[0], (x[1], dealWithNaN(x, 2, 0)))) \
        .leftOuterJoin(uid_avg_stars).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], dealWithNaN(x, 3, 3)))) \
        .leftOuterJoin(uid_var_stars).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], x[3], dealWithNaN(x, 4, 0)))) \
        .leftOuterJoin(uid_num_friends).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], x[3], x[4], dealWithNaN(x, 5, 1)))) \
        .leftOuterJoin(uid_num_iteractions) \
        .map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][1][0], x[1][1][1], x[1][1][2])) \
        .map(lambda x: (x[0], (x[1], x[2], x[3], x[4], x[5], dealWithNaN(x, 6, 0), dealWithNaN(x, 7, 0), dealWithNaN(x, 8, 0)))) \
        .leftOuterJoin(uid_num_compliment) \
        .map(lambda x: (x[0], (x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5], x[1][0][6], x[1][0][7],
                        x[1][1][0], x[1][1][1], x[1][1][2], x[1][1][3], x[1][1][4], x[1][1][5], x[1][1][6], x[1][1][7], x[1][1][8], x[1][1][9], x[1][1][10]))) \
        .leftOuterJoin(uid_yelp_since) \
        .map(lambda x: (x[0], (x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5], x[1][0][6], x[1][0][7],
                        x[1][0][8], x[1][0][9], x[1][0][10], x[1][0][11], x[1][0][12], x[1][0][13], x[1][0][14], x[1][0][15], x[1][0][16], x[1][0][17], x[1][0][18], x[1][1])))

    # print(uid_final_train.take(3))
    # exit()
    # merge them together
    # final format: uid, bid, num_photo, bid_num_review, bid_avg_stars, bid_var, bid_open, bid_state,
    # uid_num_fans, uid_num_reviews, uid_avg_stars, uid_var, num_friends,
    # x["useful"], x["funny"], x["cool"], \
    #     x["compliment_hot"], x["compliment_more"], x["compliment_profile"], x["compliment_cute"], x["compliment_list"],
    #     x["compliment_note"], x["compliment_plain"], x["compliment_cool"], \
    #     x["compliment_funny"], x["compliment_writer"], x["compliment_photos"], x["yelping_since"]
    uid_bid_star_final = uid_bid_star.map(lambda x: (x[1], (x[0], x[2]))).leftOuterJoin(bid_final_train) \
        .map(lambda x: (x[1][0][0], (x[0], x[1][1][0], x[1][1][1], x[1][1][2], x[1][1][3], x[1][1][4], x[1][1][5], x[1][0][1]))) \
        .leftOuterJoin(uid_final_train)  \
        .map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5], x[1][0][6], x[1][1][0], x[1][1][1], x[1][1][2], x[1][1][3],
        x[1][1][4], x[1][1][5], x[1][1][6], x[1][1][7], x[1][1][8], x[1][1][9], x[1][1][10], x[1][1][11], x[1][1][12], x[1][1][13], x[1][1][14],
        x[1][1][15],x[1][1][16],x[1][1][17],x[1][1][18], x[1][1][19], x[1][0][7])).collect()

    # print(uid_bid_star_final.take(2))
    # exit()
    # prepare for the machine learning train data
    # print("NO PROB BEFORE TRAIN")
    df_train_features_add = pd.DataFrame(
        {"uid": [p[0]+"_uid" for p in uid_bid_star_final],"bid": [p[1]+"_bid" for p in uid_bid_star_final],
        "num_photo": [p[2] for p in uid_bid_star_final], "bid_num_review": [p[3] for p in uid_bid_star_final],
        "bid_avg_stars": [p[4] for p in uid_bid_star_final],
        "bid_open": [p[6] for p in uid_bid_star_final], "bid_city": [p[7] for p in uid_bid_star_final],
        "uid_num_fans": [p[8] for p in uid_bid_star_final],
        "uid_num_reviews": [p[9] for p in uid_bid_star_final], "uid_avg_stars": [p[10] for p in uid_bid_star_final],
        "uid_num_friends": [p[12] for p in uid_bid_star_final],
        "uid_interactions1": [p[13] for p in uid_bid_star_final],"uid_interactions2": [p[14] for p in uid_bid_star_final],
        "uid_interactions3": [p[15] for p in uid_bid_star_final],
        "uid_compliment1": [p[16] for p in uid_bid_star_final], "uid_compliment2": [p[17] for p in uid_bid_star_final],
        "uid_compliment3": [p[18] for p in uid_bid_star_final],"uid_compliment4": [p[19] for p in uid_bid_star_final],
        "uid_compliment5": [p[20] for p in uid_bid_star_final],"uid_compliment6": [p[21] for p in uid_bid_star_final],
        "uid_compliment7": [p[22] for p in uid_bid_star_final],"uid_compliment8": [p[23] for p in uid_bid_star_final],
        "uid_compliment9": [p[24] for p in uid_bid_star_final],"uid_compliment10": [p[25] for p in uid_bid_star_final],
        "uid_compliment11": [p[26] for p in uid_bid_star_final],
        "uid_yelp_since": [p[27] for p in uid_bid_star_final]})
    df_train_features_add = pd.get_dummies(df_train_features_add,columns=["bid_city"]).drop(["bid_open"], axis=1)
    # print(df_train_features.columns)
    # exit()
    '''
    df_train_features = df_train_features_add.merge(df_features, left_on="uid", right_on="id", how="left")\
        .merge(df_features, left_on="bid", right_on="id", how="left", suffixes= ("_1","_2"))\
    .drop(["bid","uid",'id_1', 'id_2'], axis=1).fillna(0)
    '''
    # catfeatures = pd.read_csv("CatEmbeddingFeatures.csv")


    business_rdd_cf = sc.textFile(business).map(lambda line: json.loads(line)).map(
            lambda x: (x['business_id'], x['categories']))
    
    remaining_business = business_rdd_cf.partitionBy(10, PartHash) \
            .map(lambda x: (x[0]+"_bid",x[1])) \
            .map(lambda x: (x[0], x[1].replace("&"," ").replace("/"," ").replace("(","").replace(")","").replace("  ","").split(",") if x[1] is not None else ""))\
            .map(lambda x: (x[0], [h.strip().lower() for h in x[1]])).flatMap(lambda x: map(lambda type: (x[0], type), x[1])) \
            .map(lambda x: (md5(x[1].encode(encoding='UTF-8')).hexdigest(),x[0])).collect()
    
    # transfer to data frame
    type_res = [i[0] for i in remaining_business]
    id_cf = [i[1] for i in remaining_business]
    df_conn1 = pd.DataFrame({"type": type_res, "bid_cat": id_cf})
    
    
    add_info = list(set(list(df_conn1["bid_cat"])))
    df_conn_add = pd.DataFrame({"type": add_info, "bid_cat": add_info})
    df_conn = pd.concat([df_conn1,df_conn_add])
    
    
    # merge two datasets
    catfeatures = df_conn.merge(df_features, left_on="type", right_on="id", how="left").drop(["type","id"], axis=1).fillna(0).\
        groupby("bid_cat").mean()
    catfeatures.reset_index(inplace=True)
    # print(catfeatures[:10])
    # exit()
    # df_xg_train = pd.read_csv("df_xg_train.csv")
    df_train_features = df_train_features_add \
        .merge(df_features, left_on="uid", right_on="id", how="left") \
        .drop(["uid","id"], axis=1) \
        .merge(catfeatures, left_on="bid", right_on="bid_cat", how="left") \
        .drop(["bid", "bid_cat"], axis=1).fillna(0)
    df_train_target = pd.DataFrame({"score": [p[28] for p in uid_bid_star_final]})

    '''
    df_train_features = df_train_features_add.merge(df_features, left_on="bid", right_on="id", how="left") \
        .drop(["bid", "uid", 'id'], axis=1).fillna(0)
    '''
        # .merge(df_collFilter, left_on=["bid", "uid"], right_on=["business_id", "user_id"], how="left") \
        # .drop(["bid","uid","business_id","user_id"], axis=1).fillna(3)
        # df_train_features = df_train_features_add.merge(df_features, left_on="uid", right_on="id", how="left").drop(
        # ['uid', 'bid', 'id'], axis=1).fillna(0)
    # df_train_features.to_csv("../competition/traintey.csv",index=False)

    # scaler = preprocessing.StandardScaler()
    # scaler.fit(df_train_features)
    # pickle.dump(scaler,open("ScalerWithEmbed","wb"))
    # df_train_features_scaled = scaler.transform(df_train_features)

    # print("NO PROB BEFORE MODELING")
    # set the model
    # model = ()
    '''
    clf = GridSearchCV(model, {'max_depth': [8], 'learning_rate': [0.1,0.12],
                            "colsample_bytree": [0.3,0.4,0.5],
                            "subsample": [0.5,0.6], "alpha": [0], "random_state": [0]})
    '''
    # clf = xgb.XGBRegressor(max_depth = 6, min_child_weight = 2, eta=0.05, subsample=0.8, colsample_bytree = 0.8)
    # clf = lgb.LGBMRegressor(boosting_type='gbdt', objective='mse', num_leaves=100, max_depth=7, learning_rate=0.06, n_estimators=400)
    # clf = xgb.XGBRegressor(max_depth = 7, subsample=0.85, colsample_bytree=0.3, n_estimators = 90, learning_rate=0.1)
    clf = CatBoostRegressor(n_estimators=1200, max_depth=7,  boosting_type="Plain",learning_rate=0.06, random_state=0)
    score = cross_val_score(clf, df_train_features, df_train_target, scoring='neg_root_mean_squared_error', cv=10)
    clf.fit(df_train_features, df_train_target)
    print(score)
    # clf.fit(df_train_features,df_train_target)
    # marshal.dump(clf,"xgboost")
    # print("\n The best parameters across ALL searched params:\n", clf.best_params_)

    print("training_RMSE:", mean_squared_error(y_true = df_train_target['score'], y_pred = clf.predict(df_train_features), squared=False))

    # exit()
    # join the data to create the test dataset
    # for bid (treat all null restaurant as 3)


    pickle.dump(clf,open("CatGBMRegWithEmbedV4Long", "wb"))

    # clf = pickle.load(open("CatGBMRegWithEmbedV1","rb"))

    # print("NO PROB BEFORE TEST")
    bid_final_test = bid_test.leftOuterJoin(restaurant_photo).map(lambda x: (x[0], x[1][1])).map(
        lambda x: (x[0], dealWithNaN(x, 1, 0))) \
        .leftOuterJoin(bid_num_review).map(lambda x: (x[0], x[1][0], x[1][1])).map(
        lambda x: (x[0], (x[1], dealWithNaN(x, 2, 0)))) \
        .leftOuterJoin(bid_avg_stars).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], dealWithNaN(x, 3, 3)))) \
        .leftOuterJoin(bid_var_stars).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], x[3], dealWithNaN(x, 4, 0)))) \
        .leftOuterJoin(bid_open_state).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][1][0], x[1][1][1])) \
        .map(lambda x: (x[0], (x[1], x[2], x[3], x[4], dealWithNaN(x, 5, "NULL"), dealWithNaN(x, 6, 1))))
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
        .map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][1][0], x[1][1][1], x[1][1][2])) \
        .map(lambda x: (x[0], (x[1], x[2], x[3], x[4], x[5], dealWithNaN(x, 6, 0), dealWithNaN(x, 7, 0), dealWithNaN(x, 8, 0)))) \
        .leftOuterJoin(uid_num_compliment) \
        .map(lambda x: (x[0], (x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5], x[1][0][6], x[1][0][7],
        x[1][1][0], x[1][1][1], x[1][1][2], x[1][1][3], x[1][1][4], x[1][1][5], x[1][1][6], x[1][1][7], x[1][1][8],
        x[1][1][9], x[1][1][10]))) \
        .leftOuterJoin(uid_yelp_since) \
        .map(lambda x: (x[0], (x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5], x[1][0][6], x[1][0][7],
        x[1][0][8], x[1][0][9], x[1][0][10], x[1][0][11], x[1][0][12], x[1][0][13], x[1][0][14], x[1][0][15],
        x[1][0][16], x[1][0][17], x[1][0][18], x[1][1])))

    # final format: uid, bid, num_photo, bid_num_review, bid_avg_stars, uid_num_fans, uid_num_reviews, uid_avg_stars, actual_stars
    uid_bid_star_final_test = uid_bid_need.map(lambda x: (x[1], (x[0]))).leftOuterJoin(bid_final_test) \
        .map(lambda x: (x[1][0], (x[0], x[1][1][0], x[1][1][1], x[1][1][2], x[1][1][3], x[1][1][4], x[1][1][5]))) \
        .leftOuterJoin(uid_final_test)  \
        .map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5], x[1][0][6], x[1][1][0], x[1][1][1], x[1][1][2], x[1][1][3],
        x[1][1][4], x[1][1][5], x[1][1][6], x[1][1][7], x[1][1][8], x[1][1][9], x[1][1][10], x[1][1][11], x[1][1][12], x[1][1][13], x[1][1][14],
        x[1][1][15],x[1][1][16],x[1][1][17],x[1][1][18], x[1][1][19])).collect()

    # print(uid_bid_star_final_test)
    # exit()
    # prepare for the machine learning train data
    df_test_features_add = pd.DataFrame({"uid": [p[0]+"_uid" for p in uid_bid_star_final_test],"bid": [p[1]+"_bid" for p in uid_bid_star_final_test],
        "num_photo": [p[2] for p in uid_bid_star_final_test], "bid_num_review": [p[3] for p in uid_bid_star_final_test],
        "bid_avg_stars": [p[4] for p in uid_bid_star_final_test], 
        "bid_open": [p[6] for p in uid_bid_star_final_test], "bid_city": [p[7] for p in uid_bid_star_final_test],
        "uid_num_fans": [p[8] for p in uid_bid_star_final_test],
        "uid_num_reviews": [p[9] for p in uid_bid_star_final_test], "uid_avg_stars": [p[10] for p in uid_bid_star_final_test],
        "uid_num_friends": [p[12] for p in uid_bid_star_final_test],
        "uid_interactions1": [p[13] for p in uid_bid_star_final_test],"uid_interactions2": [p[14] for p in uid_bid_star_final_test],
        "uid_interactions3": [p[15] for p in uid_bid_star_final_test],
        "uid_compliment1": [p[16] for p in uid_bid_star_final_test], "uid_compliment2": [p[17] for p in uid_bid_star_final_test],
        "uid_compliment3": [p[18] for p in uid_bid_star_final_test],"uid_compliment4": [p[19] for p in uid_bid_star_final_test],
        "uid_compliment5": [p[20] for p in uid_bid_star_final_test],"uid_compliment6": [p[21] for p in uid_bid_star_final_test],
        "uid_compliment7": [p[22] for p in uid_bid_star_final_test],"uid_compliment8": [p[23] for p in uid_bid_star_final_test],
        "uid_compliment9": [p[24] for p in uid_bid_star_final_test],"uid_compliment10": [p[25] for p in uid_bid_star_final_test],
        "uid_compliment11": [p[26] for p in uid_bid_star_final_test],
        "uid_yelp_since": [p[27] for p in uid_bid_star_final_test]})
    df_test_features_add = pd.get_dummies(df_test_features_add, columns=["bid_city"]).drop(["bid_open"],axis=1)
    '''
    df_test_features = df_test_features_add.merge(df_features, left_on="uid", right_on="id", how="left").merge(
        df_features, left_on="bid", right_on="id", how="left", suffixes=("_1", "_2")).drop(
        ['uid', 'bid', 'id_1', 'id_2'], axis=1).fillna(0)
    df_train_features = df_train_features_add.merge(df_features, left_on="uid", right_on="id", how="left") \
        .drop(["uid", 'id'], axis=1).merge(catfeatures, left_on="bid", right_on="bid_cat", how="left") \
        .drop(["bid", "bid_cat"], axis=1).fillna(0)
    '''
    # df_xg_test = pd.read_csv("xg_final.csv")
    df_test_features = df_test_features_add \
        .merge(df_features, left_on="uid", right_on="id", how="left")\
        .drop(['uid', 'id'], axis=1) \
        .merge(catfeatures, left_on="bid", right_on="bid_cat", how="left") \
        .drop(["bid", "bid_cat"], axis=1).fillna(0)

    '''
    df_test_features = df_test_features_add.merge(df_features, left_on="bid", right_on="id", how="left").drop(
        ['uid', 'bid', 'id'], axis=1).fillna(0)
    '''
    # df_test_features = df_test_features_add.merge(df_features, left_on="uid", right_on="id", how="left").drop(
        # ['uid', 'bid', 'id'], axis=1).fillna(0)
    # df_test_target = pd.DataFrame({"score": [p[8] for p in uid_bid_star_final_test]})
    # df_test_features_scaled = scaler.transform(df_test_features)

    y_pred = clf.predict(df_test_features)
    '''
    df_final = pd.DataFrame(
        {"user_id": [p[0] for p in uid_bid_star_final_test], "business_id": [p[1] for p in uid_bid_star_final_test], "num_photo": [p[2] for p in uid_bid_star_final_test], "bid_num_review": [p[3] for p in uid_bid_star_final_test],
        "bid_avg_stars": [p[4] for p in uid_bid_star_final_test], "bid_var": [p[5] for p in uid_bid_star_final_test],"uid_num_fans": [p[6] for p in uid_bid_star_final_test],
        "uid_num_reviews": [p[7] for p in uid_bid_star_final_test], "uid_avg_stars": [p[8] for p in uid_bid_star_final_test],"uid_var":[p[9] for p in uid_bid_star_final_test],
        "uid_num_friends":[p[10] for p in uid_bid_star_final_test], "uid_num_interactions": [p[11] for p in uid_bid_star_final_test],"uid_num_compliment": [p[12] for p in uid_bid_star_final_test], "uid_yelp_since":[p[13] for p in uid_bid_star_final_test],
        "prediction": y_pred})
    '''
    df_final = pd.DataFrame(
        {"user_id": [p[0] for p in uid_bid_star_final_test], "business_id": [p[1] for p in uid_bid_star_final_test],
        "prediction": y_pred})

    # print("testing_RMSE:",mean_squared_error(y_true=df_test_target['score'], y_pred=clf.predict(df_test_features_scaled),squared=False))
    end_time = time.time()
    duration = end_time - start_time
    print("Duration:", duration)
    df_final.to_csv(output_filepath, index=False)


    df3 = pd.read_csv(test_filepath)

    df = df3.merge(df_final, on=["user_id", "business_id"])
    df["diff"] = abs(df["stars"] - df["prediction"])
    df["diff_2"] = [i ** 2 for i in df["diff"]]
    print("Here", (sum(list(df["diff_2"])) / len(list(df["diff_2"]))) ** 0.5)

    # df_final.to_csv(output_filepath,index=False)
    # rmse: 0.974449082769936
    # rmse: 0.9740866070958515 with new embedding features
    # Here 0.950877063349136
