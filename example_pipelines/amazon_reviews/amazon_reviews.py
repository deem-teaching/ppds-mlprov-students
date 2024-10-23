import datetime

import numpy as np
import pandas as pd
from faker import Faker
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize


def load_data():
    print("  Data access - Loading all four input relations from github")
    base_url = 'https://github.com/schelterlabs/arguseyes/raw/freamon/arguseyes/example_pipelines/datasets/freamon'

    reviews = pd.read_csv(f'{base_url}/reviews.csv.gz', compression='gzip', index_col=0)
    ratings = pd.read_csv(f'{base_url}/ratings.csv', index_col=0)
    products = pd.read_csv(f'{base_url}/products.csv', index_col=0)
    categories = pd.read_csv(f'{base_url}/categories.csv', index_col=0)

    return reviews, ratings, products, categories


def random_subset(arr):
    size = np.random.randint(low=1, high=len(arr) + 1)
    choice = np.random.choice(arr, size=size, replace=False)
    return [str(item) for item in choice]


def integrate_data(reviews, ratings, products, categories, fake):
    start_date = fake.date_between(start_date=datetime.date(year=2011, month=1, day=1),
                                   end_date=datetime.date(year=2013, month=6, day=1))

    print(f"  Data Integration - Dropping reviews written before {start_date.strftime('%Y-%m-%d')}")
    reviews = reviews[reviews.review_date >= start_date.strftime('%Y-%m-%d')]

    reviews_with_ratings = reviews.merge(ratings, on='review_id')
    products_with_categories = products.merge(left_on='category_id', right_on='id', right=categories)

    random_categories = random_subset(list(categories.category))
    print(f"  Data Integration - restricting products to the following categories {random_categories}")
    products_with_categories = products_with_categories[products_with_categories.category.isin(random_categories)]

    reviews_with_products_and_ratings = reviews_with_ratings.merge(products_with_categories, on='product_id')

    print(f"  Data Integration - joined reviews, ratings, products & categories")
    return reviews_with_products_and_ratings


def compute_feature_and_label_data(reviews_with_products_and_ratings, final_columns, fake):
    reviews_with_products_and_ratings['product_title'] = \
        reviews_with_products_and_ratings['product_title'].fillna(value='')

    reviews_with_products_and_ratings['review_headline'] = \
        reviews_with_products_and_ratings['review_headline'].fillna(value='')

    reviews_with_products_and_ratings['review_body'] = \
        reviews_with_products_and_ratings['review_body'].fillna(value='')

    num_text_columns = np.random.randint(low=1, high=4)
    random_text_columns = np.random.choice(['product_title', 'review_headline', 'review_body'],
                                           size=num_text_columns, replace=False)

    print(f"  Data preparation - using columns {random_text_columns} as textual feature ")
    reviews_with_products_and_ratings['text'] = ' '
    for text_column in random_text_columns:
        reviews_with_products_and_ratings['text'] = reviews_with_products_and_ratings['text'] + ' ' \
                                                    + reviews_with_products_and_ratings[text_column]

    reviews_with_products_and_ratings['is_helpful'] = reviews_with_products_and_ratings['helpful_votes'] > 0

    print(f"  Projecting data to the feature and label columns {final_columns}")
    projected_reviews = reviews_with_products_and_ratings[final_columns]

    split_date = fake.date_between(start_date=datetime.date(year=2013, month=12, day=1),
                                   end_date=datetime.date(year=2015, month=1, day=1))

    print(f"  Data preparation - temporal train/test split based on date {split_date}")
    train_data = projected_reviews[projected_reviews.review_date <= split_date.strftime('%Y-%m-%d')]
    train_labels = label_binarize(train_data['is_helpful'], classes=[True, False]).ravel()

    test_data = projected_reviews[projected_reviews.review_date > split_date.strftime('%Y-%m-%d')]
    test_labels = label_binarize(test_data['is_helpful'], classes=[True, False]).ravel()

    return train_data, train_labels, test_data, test_labels


def define_model(numerical_columns, categorical_columns):
    print(f"  Feature encoding - Setting up feature transformations")
    feature_transformation = ColumnTransformer(transformers=[
        ('numerical_features', StandardScaler(), numerical_columns),
        ('categorical_features', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('textual_features', HashingVectorizer(ngram_range=(1, 3), n_features=100), 'text'),
    ], remainder="drop")

    print(f"  Modeling - defining a logistic regression model")
    sklearn_model = Pipeline([
        ('features', feature_transformation),
        ('learner', SGDClassifier(loss='log_loss', penalty='l1', max_iter=1000))])

    return sklearn_model


def execute_pipeline(seed):
    fake = Faker()
    fake.seed_instance(seed)
    np.random.seed(seed)

    print('---------------------------------------------------------------------')
    print(f'Executing pipeline with seed {seed}')
    print('---------------------------------------------------------------------')

    numerical_columns = random_subset(['total_votes', 'star_rating'])
    categorical_columns = random_subset(['customer_id', 'product_id', 'vine', 'category'])
    final_columns = numerical_columns + categorical_columns + ['text', 'is_helpful', 'review_date']

    reviews, ratings, products, categories = load_data()

    integrated_data = integrate_data(reviews, ratings, products, categories, fake)
    train_data, train_labels, test_data, test_labels = \
        compute_feature_and_label_data(integrated_data, final_columns, fake)

    sklearn_model = define_model(numerical_columns, categorical_columns)

    model = sklearn_model.fit(train_data, train_labels)

    print('---------------------------------------------------------------------')
    print('Train accuracy', model.score(train_data, train_labels))
    print('Test accuracy', model.score(test_data, test_labels))
    print('---------------------------------------------------------------------\n\n')


execute_pipeline(42)
