"""
Project-wide constants.

This module defines constants that are part of the data schema or project
structure but are not intended to be tuned as hyperparameters.
"""

# --- FILENAMES ---
TRAIN_FILENAME = "stage1_public_train.csv"
TEST_FILENAME = "stage1_public_test.csv"
USER_DATA_FILENAME = "stage1_public_users.csv"
BOOK_DATA_FILENAME = "stage1_public_books.csv"
BOOK_GENRES_FILENAME = "stage1_public_book_genres.csv"
GENRES_FILENAME = "stage1_public_genres.csv"
SUBMISSION_FILENAME = "submission.csv"

# --- COLUMN NAMES ---
# Main columns
COL_USER_ID = "user_id"
COL_BOOK_ID = "book_id"
COL_TARGET = "rating"
COL_SOURCE = "source"
COL_PREDICTION = "rating_predict"

# Feature columns (newly created)
F_USER_MEAN_RATING = "user_mean_rating"
F_USER_RATINGS_COUNT = "user_ratings_count"
F_BOOK_MEAN_RATING = "book_mean_rating"
F_BOOK_RATINGS_COUNT = "book_ratings_count"
F_AUTHOR_MEAN_RATING = "author_mean_rating"
F_BOOK_GENRES_COUNT = "book_genres_count"

# Metadata columns from raw data
COL_GENDER = "gender"
COL_AGE = "age"
COL_AUTHOR_ID = "author_id"
COL_PUBLICATION_YEAR = "publication_year"
COL_LANGUAGE = "language"
COL_PUBLISHER = "publisher"
COL_AVG_RATING = "avg_rating"
COL_GENRE_ID = "genre_id"


# --- VALUES ---
VAL_SOURCE_TRAIN = "train"
VAL_SOURCE_TEST = "test"

# --- MAGIC NUMBERS ---
MISSING_CAT_VALUE = "-1"
MISSING_NUM_VALUE = -1
PREDICTION_MIN_VALUE = 0
PREDICTION_MAX_VALUE = 10
