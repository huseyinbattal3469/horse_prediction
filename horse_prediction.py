# # https://www.kaggle.com/competitions/playground-series-s4e5/overview
# Made by Hüseyin Battal.
# GitHub: https://github.com/Husolm
# LinkedIn: https://www.linkedin.com/in/huseyin-battal/


import warnings
from required_functions import *


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)

df_train = pd.read_csv("Kaggle/horse_prediction/train.csv")
df_test = pd.read_csv("Kaggle/horse_prediction/test.csv")
df_ = pd.concat([df_train, df_test],axis=0)

df = df_.copy()

# EDA

check_df(df)

cat_cols, num_cols, cat_but_car_cols = grab_col_names(df)

df["lesion_1"].nunique()
df["lesion_2"].nunique()
df["lesion_3"].nunique()

for col in num_cols:
    num_summary(df, col,True)

for col in cat_cols:
    cat_summary(df, col, True)

# Outlier Analyze
for col in num_cols:
    result = check_outlier(df,col)
    print(col, result)
    print(show_outliers(df, col))
    sns.boxplot(x=df[col])
    plt.show(block=True)

# Outlier optimization
dff = df.copy()

# exception for ID
for col in num_cols[1:]:
    replace_with_thresholds(dff, col, low_threshold=True)

for col in num_cols:
    result = check_outlier(dff,col)
    print(col, result)
    print(show_outliers(dff, col))
    sns.boxplot(x=dff[col])
    plt.show(block=True)


check_df(dff)

# Empty Value Analyze
missing_values_table(dff) # Only the dependant value

# Scaling and Labelling
dff2 = dff.copy()
dff2.drop("id",axis=1,inplace=True)
dff2.drop("hospital_number",axis=1,inplace=True)
cat_cols, num_cols, cat_but_car = grab_col_names(dff2)
binary_cols = [col for col in cat_cols if dff2[col].nunique() == 2]
cat_cols = [col for col in cat_cols if col not in binary_cols]

le_result = LabelEncoder()

for col in binary_cols:
    dff2 = label_encoder(dff2, col)

for col in num_cols:
    dff2 = robust_scaler(dff2,col)

for col in [col for col in cat_cols if col not in "outcome"]:
    dff2 = label_encoder(dff2, col)

dff2_train = dff2[~dff2["outcome"].isnull()]
dff2_train["outcome"] = le_result.fit_transform(dff2_train["outcome"])
dff2_test = dff2[dff2["outcome"].isnull()]

X = dff2_train.drop("outcome",axis=1)
y = dff2_train["outcome"]
X_test_df = dff2_test.drop("outcome",axis=1)

best_models = hyperparameter_multiclass_optimization(X,y,cv=5)
final_model = voting_model(best_models, X, y,5,is_multiclass=True)
final_model.fit(X, y)
y_test_predicted = final_model.predict(X_test_df)
y_test_predicted = le_result.inverse_transform(y_test_predicted)
df_result = pd.DataFrame({"id":df_test.id,"outcome":y_test_predicted})
df_result
df_result.to_csv("horse_result.csv",index=None)















