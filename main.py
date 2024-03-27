import pandas as pd
from IPython.display import display
from tabulate import tabulate
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


number_of_columns_to_display = 60
number_of_characters_to_display = 500

pd.options.display.max_columns = number_of_columns_to_display
pd.options.display.max_colwidth = number_of_characters_to_display

# Getting each json file as a pandas dataframe.
businesses = pd.read_json("yelp_business.json", lines=True)
reviews = pd.read_json("yelp_review.json", lines=True)
users = pd.read_json("yelp_user.json", lines=True)
checkins = pd.read_json("yelp_checkin.json", lines=True)
tips = pd.read_json("yelp_tip.json", lines=True)
photos = pd.read_json("yelp_photo.json", lines=True)

# Merging the dataframes into one big dataframe.
df = pd.merge(businesses, reviews, how="left", on="business_id")
df = pd.merge(df, users, how="left", on="business_id")
df = pd.merge(df, checkins, how="left", on="business_id")
df = pd.merge(df, tips, how="left", on="business_id")
df = pd.merge(df, photos, how="left", on="business_id")

# Removing Columns that are not numeric or binary.
features_to_remove = [
    "address",
    "attributes",
    "business_id",
    "categories",
    "city",
    "hours",
    "is_open",
    "latitude",
    "longitude",
    "name",
    "neighborhood",
    "postal_code",
    "state",
    "time",
]
df.drop(labels=features_to_remove, axis=1, inplace=True)

# Checking for columns with missing values and filling them with binary 0.
# print(df.isna().any())
df.fillna(
    {
        "weekday_checkins": 0,
        "weekend_checkins": 0,
        "average_tip_length": 0,
        "number_tips": 0,
        "average_caption_length": 0,
        "number_pics": 0,
    },
    inplace=True,
)

# Confirming missing values have been filled.
# print(df.isna().any())

# Checking for strong correlations.
"""
# Plotting average_review_sentiment against stars
plt.scatter(df["average_review_sentiment"], df["stars"], alpha=0.1)
plt.xlabel("average_review_sentiment")
plt.ylabel("Yelp Rating")
plt.show()

# Plotting average_review_length against stars here.
plt.scatter(df["average_review_length"], df["stars"], alpha=0.1)
plt.xlabel("average_review_length")
plt.ylabel("Yelp Rating")
plt.show()

# plotting average_review_age against stars here.
plt.scatter(df["average_review_age"], df["stars"], alpha=0.1)
plt.xlabel("average_review_age")
plt.ylabel("Yelp Rating")
plt.show()

# plotting number_funny_votes against stars here.
plt.scatter(df["number_funny_votes"], df["stars"], alpha=0.1)
plt.xlabel("number_funny_votes")
plt.ylabel("Yelp Rating")
plt.show()
"""

# Preparing data for linear regression.
ratings = df["stars"]
features = df[["average_review_length", "average_review_age"]]


# Getting training data, testing data, training-dependent variable, testing-dependent variable.
X_train, X_test, y_train, y_test = train_test_split(
    features, ratings, test_size=0.2, random_state=1
)

# Instancing a Linear Regression model.
model = LinearRegression()

# Training the model using the fit method.
model.fit(X_train, y_train)

# Seeing what proportion of variance of stars could be explained by the features.
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

# About 8% for each feature.
