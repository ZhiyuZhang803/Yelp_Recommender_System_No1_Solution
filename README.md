# Yelp_Recommender_System_No1_Solution

- Introduction: This repo contains all files needed for building a recommender system based on 2019 Yelp Challenge Datasets. This is the No.1 solution in USC Viterbi Data Mining Competition. 

- Goal: With existing business and user information, we try to build a model to predict the ratings of new business-user pairs.

### Workflow of Model:

- **Step1**: With given business-business relationships, business-user relationships, and business-categories relationships,
we build a graph that makes connection among them. Then, graph embedding technique is used to construct 200 dimension numeric
vectors that can represent the relationships between them.

(File name: 'GraphEmbedding.py'; Output: 'VectorizedFeatures.csv')

- **Step2**: Categories of restaurants shouldn't be viewed as separated features. Therefore, I add them back to the business
vectors as side information to enhance the expression of individual business. 

(NOTE: Because the memory limit of Vocareum, I incorporate this step into training phase files.)

- **Step3**: Two machine learning models are trained based on the embedding factors, business features, and user features.

Model 1: XGBOOST
(File name: 'competition_XGBOOST.py'; Output: 'XGBoostRegWithEmbedV4Long' - pickle file)

Model 2: CatBOOST
(File name: 'competition_CatBOOST.py'; Output: 'CatGBMRegWithEmbedV4Long' - pickle file)

- **Step4**: Outputs of the models are combined. We replace prediction scores that >5 as 5 and prediction scores that <1 as 1
before combination and after combination. Then we get our final results!

### Details of Files:


### Special Notes:
- Please do not copy the codes directly with any purpose.
- In order to protect the copyright, we do not provide original datasets.
- If you want to get the pickle files that are used to generate final models, please contact the author.
