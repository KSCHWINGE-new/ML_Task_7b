#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from flask import Flask, request, jsonify
from scipy.stats import entropy

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = df['week'].apply(lambda x: f'2023-W{x}-1')
    df['date'] = pd.to_datetime(df['date'], format='%G-W%V-%u')
    return df

# Optional: Remove infrequent products.
def remove_infrequent_products(data, n):
    product_counts = data['sku_id'].value_counts()
    infrequent_products = product_counts[product_counts < n].index
    filtered_data = data[~data['sku_id'].isin(infrequent_products)]
    return filtered_data

def aggregate_data(data):
    aggregated_data = data.groupby(['date', 'sku_id', 'fulfilment_location_id']).agg({'order_quantity': 'sum'}).reset_index()
    sku_id_map = {val: idx for idx, val in enumerate(aggregated_data['sku_id'].unique())}
    fulfilment_location_id_map = {val: idx for idx, val in enumerate(aggregated_data['fulfilment_location_id'].unique())}
    aggregated_data['sku_id'] = aggregated_data['sku_id'].map(sku_id_map)
    aggregated_data['fulfilment_location_id'] = aggregated_data['fulfilment_location_id'].map(fulfilment_location_id_map)
    return aggregated_data, sku_id_map, fulfilment_location_id_map

def expand_data(aggregated_data):
    full_date_range = pd.date_range(start=aggregated_data['date'].min(), end=aggregated_data['date'].max(), freq='W-MON')
    product_location_pairs = aggregated_data[['sku_id', 'fulfilment_location_id']].drop_duplicates()
    expanded_data = pd.MultiIndex.from_product([full_date_range, product_location_pairs['sku_id'], product_location_pairs['fulfilment_location_id']], names=['date', 'sku_id', 'fulfilment_location_id']).to_frame(index=False)
    aggregated_data = expanded_data.merge(aggregated_data, on=['date', 'sku_id', 'fulfilment_location_id'], how='left').fillna({'order_quantity': 0})
    return aggregated_data

def week_of_month(dt):
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()
    return int(np.ceil(adjusted_dom / 7.0))

def preprocess_data(aggregated_data):
    aggregated_data['week_of_month'] = aggregated_data['date'].apply(week_of_month)
    aggregated_data = aggregated_data.drop(columns=['date'])
    return aggregated_data

def predict_demand(date, product_id, location_id, sku_id_map, fulfilment_location_id_map, clf, reg):
    if product_id not in sku_id_map or location_id not in fulfilment_location_id_map:
        raise ValueError("Provided product_id or location_id is not in the map. Set 'remove_infrequent_products' threshold lower")

    encoded_product_id = sku_id_map[product_id]
    encoded_location_id = fulfilment_location_id_map[location_id]
    week = week_of_month(pd.to_datetime(date))
    input_data = pd.DataFrame({
        'sku_id': [encoded_product_id],
        'fulfilment_location_id': [encoded_location_id],
        'week_of_month': [week]
    }, columns=['sku_id', 'fulfilment_location_id', 'week_of_month'])
    
    probabilities = clf.predict_proba(input_data)[0]
    is_non_zero = clf.predict(input_data)[0]

    # Entropy used as a measure of uncertainty 
    classification_uncertainty = entropy(probabilities)

    if is_non_zero == 0:
        prediction = 0.0
        uncertainty = classification_uncertainty
    else:
        prediction = reg.predict(input_data)[0]
        preds = np.array([estimator.predict(input_data)[0] for estimator in reg.estimators_])
        regression_uncertainty = np.std(preds)
        # Combine uncertainties in quadrature 
        uncertainty = np.sqrt(classification_uncertainty**2 + regression_uncertainty**2)

    return prediction, uncertainty


def main(file_path):
    df = load_data(file_path)
    print("Data loaded.")
    f_df = remove_infrequent_products(df, 10)
    print("Infrequent products removed.")
    aggregated_data, sku_id_map, fulfilment_location_id_map = aggregate_data(f_df)
    print("Data aggregated.")
    aggregated_data = expand_data(aggregated_data)
    print("Data expanded.")
    data = preprocess_data(aggregated_data)
    X = data.drop(columns=['order_quantity'])
    y = data['order_quantity']
    print("Data preprocessed.")
    y_binary = (y > 0).astype(int)
    clf = RandomForestClassifier(n_estimators=700, random_state=44)
    clf.fit(X, y_binary)
    print("Classification model trained.")
    X_nonzero = X[y > 0]
    y_nonzero = y[y > 0]
    reg = RandomForestRegressor(n_estimators=700, random_state=44)
    reg.fit(X_nonzero, y_nonzero)
    print("Regression model trained.")

    return clf, reg, sku_id_map, fulfilment_location_id_map

app = Flask(__name__)

clf, reg, sku_id_map, fulfilment_location_id_map = main('/Users/kolyaschwinge/downloads/demand_forecasting_dataset.csv')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    date = data.get('date')
    product_id = data.get('product_id')
    location_id = data.get('location_id')
    
    try:
        predicted_demand, uncertainty = predict_demand(date, product_id, location_id, sku_id_map, fulfilment_location_id_map, clf, reg)
        return jsonify({
            'predicted_demand': predicted_demand,
            'uncertainty': uncertainty
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)
    print("Flask server started.")
