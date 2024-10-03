import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from surprise import SVD, Reader, Dataset, KNNBasic
from surprise.model_selection import GridSearchCV
from surprise.model_selection import train_test_split


# takes a file and returns it as a dataframe
def process_data_as_df(file_path, delimiter, columns):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # splits each line into a list of values
    data = [line.strip().split(delimiter) for line in lines]

    df = pd.DataFrame(data)
    if columns:
        df.columns=columns

    return df

# returns ratings data. drops the timestamp column and converts the dataframe to a Surprise data object that we can
# work with
def ratings_data_preprocessed():
    ratings_data = process_data_as_df('ml-100k/u.data', '\t', None)
    ratings_data.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings_data.drop(["timestamp"], axis=1, inplace=True)

    reader = Reader(rating_scale=(1, 5))  # Specify the rating scale
    return Dataset.load_from_df(ratings_data[ratings_data.columns], reader)

#ratings_data=ratings_data_preprocessed()


def best_SVD_parameters(data):
    param_grid = {"n_epochs": list(range(5,11)), "lr_all": [0.002, 0.003, 0.004, 0.005], "reg_all": [0.4, 0.5, 0.6]}

    eval_metrics=["rmse", "mae"]

    # cv=5 runs 5-fold cross validation, as desired
    gs = GridSearchCV(SVD, param_grid, measures=eval_metrics, cv=5)

    gs.fit(data)

    # prints best rmse and mae scores
    print(gs.best_score["rmse"])
    print(gs.best_score["mae"])

    # combination of parameters that gave the best RMSE score
    print(gs.best_params["rmse"])

    cv_results = gs.cv_results

    # Print mean and standard deviation of RMSE and MAE
    for metric in eval_metrics:
        mean_score = cv_results[f"mean_test_{metric}"]
        std_score = cv_results[f"std_test_{metric}"]
        print(f"{metric}: {mean_score.mean():.4f} ± {std_score.mean():.4f}")

# made user_based a parameter so that i can find the best parameters for user based filtering, and the best parameters
# for item based filtering
def best_KNN_parameters(data, user_based):
    param_grid = {
        'sim_options': {
            'name': ['cosine', 'pearson'],
            'user_based': [user_based]
        },
        'k': list(range(20, 30))
    }

    eval_metrics = ["rmse", "mae"]

    # cv=5 runs 5-fold cross-validation
    gs = GridSearchCV(KNNBasic, param_grid, measures=eval_metrics, cv=5)

    gs.fit(data)

    print(gs.best_score["rmse"])
    print(gs.best_score["mae"])

    print(gs.best_params["rmse"])

    cv_results = gs.cv_results

    for metric in eval_metrics:
        mean_score = cv_results[f"mean_test_{metric}"]
        std_score = cv_results[f"std_test_{metric}"]
        print(f"{metric}: {mean_score.mean():.4f} ± {std_score.mean():.4f}")

#best_SVD_parameters(ratings_data)
#best_KNN_parameters(ratings_data, True)
#best_KNN_parameters(ratings_data, False)

# finds the best linear combination of the optimal models, and returns the results
def hybrid_model(ratings_data):
    trainset, testset = train_test_split(ratings_data, test_size=0.2)

    # makes svd model with optimal parameters
    optimal_svd_model = SVD(n_epochs=10, lr_all=0.005, reg_all=0.4)
    optimal_svd_model.fit(trainset)

    # makes knn model with item based filtering and optimal parameters
    optimal_knn_item_model = KNNBasic(sim_options={'name': 'cosine', 'user_based': False}, k=29)
    optimal_knn_item_model.fit(trainset)

    optimal_knn_user_model=KNNBasic(sim_options={'name': 'pearson', 'user_based': True}, k=29)
    optimal_knn_user_model.fit(trainset)

    svd_predictions=optimal_svd_model.test(testset)
    knn_item_predictions=optimal_knn_item_model.test(testset)
    knn_user_predictions=optimal_knn_user_model.test(testset)

    svd_ratings = []
    knn_item_ratings = []
    knn_user_ratings=[]

    # predictions are Surprise objects, so we need to extract the predictions
    for svd_pred, knn_item_pred, knn_user_pred in zip(svd_predictions, knn_item_predictions, knn_user_predictions):
        svd_ratings.append(svd_pred.est)
        knn_item_ratings.append(knn_item_pred.est)
        knn_user_ratings.append(knn_user_pred.est)

    svd_ratings = np.array(svd_ratings)
    knn_item_ratings = np.array(knn_item_ratings)
    knn_user_ratings=np.array(knn_user_ratings)

    # these are the dependent variables that we will use in our regression
    X = np.column_stack((svd_ratings, knn_user_ratings))

    y = [rating for (_, _, rating) in testset]

    regression_model=LinearRegression()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rmse_scores = -cross_val_score(regression_model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
    mae_scores = -cross_val_score(regression_model, X, y, cv=kf, scoring='neg_mean_absolute_error')

    # fit the model on the entire dataset. i only did this to get a rough estimate of the coefficients of the model,
    # as i couldn't get them from the fits to the training sets
    regression_model.fit(X, y)

    # Report the results
    print(regression_model.coef_)
    print("RMSE Mean:", rmse_scores.mean())
    print("RMSE Standard Deviation:", rmse_scores.std())
    print("MAE Mean:", mae_scores.mean())
    print("MAE Standard Deviation:", mae_scores.std())

#hybrid_model(ratings_data)

