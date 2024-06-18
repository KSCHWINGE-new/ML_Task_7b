#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from prophet import Prophet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

#%%
# Load the dataset
df = pd.read_csv('/Users/kolyaschwinge/downloads/demand_forecasting_dataset.csv')
df['date'] = df['week'].apply(lambda x: f'2023-W{x}-1')
df['date'] = pd.to_datetime(df['date'], format='%G-W%V-%u')

# Optional: Removes infrequent products
def remove_infrequent_products(data, n):
    product_counts = data['sku_id'].value_counts()
    infrequent_products = product_counts[product_counts < n].index
    return data[~data['sku_id'].isin(infrequent_products)]

f_data = remove_infrequent_products(df, 10)

#%%
# My Initial EDA
plt.figure(figsize=(10, 6))
sns.histplot(f_data['order_quantity'], bins=150, kde=True)
plt.title('Distribution of Order Quantities')
plt.xlabel('Order Quantity')
plt.ylabel('Frequency')
plt.show()

num_unique_products = f_data['sku_id'].nunique()
num_unique_locations = f_data['fulfilment_location_id'].nunique()
print(f'Number of unique products: {num_unique_products}')
print(f'Number of unique locations: {num_unique_locations}')

product_order_counts = f_data['sku_id'].value_counts()
plt.figure(figsize=(10, 6))
sns.histplot(product_order_counts, bins=150, kde=True)
plt.title('Distribution of Order Counts per Product')
plt.xlabel('Order Counts')
plt.ylabel('Number of Products')
plt.show()

weekly_orders = f_data.groupby('week')['order_quantity'].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(data=weekly_orders, x='week', y='order_quantity')
plt.title('Total Order Quantity per Week')
plt.xlabel('Week')
plt.ylabel('Total Order Quantity')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='week', y='order_quantity', data=f_data)
plt.title('Distribution of Order Quantity per Week')
plt.xlabel('Week')
plt.ylabel('Order Quantity')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

df_encoded = f_data.copy()
df_encoded['sku_id'] = df_encoded['sku_id'].astype('category').cat.codes
df_encoded['fulfilment_location_id'] = df_encoded['fulfilment_location_id'].astype('category').cat.codes
df_encoded=df_encoded.drop(columns=['Unnamed: 0','week'])
corr_matrix = df_encoded.corr()

plt.figure(figsize=(12, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

location_product_counts = f_data.groupby(['fulfilment_location_id', 'sku_id']).size().reset_index(name='counts')
location_avg_product_counts = location_product_counts.groupby('fulfilment_location_id')['counts'].mean()
plt.figure(figsize=(12, 6))
location_avg_product_counts.plot(kind='bar')
plt.title('Average Number of Times a Product is Associated with an Order for Each Fulfilment Location')
plt.xlabel('Fulfilment Location ID')
plt.ylabel('Average Number of Times a Product is Associated with an Order')
plt.xticks(rotation=45)
plt.show()

#%%
# Data Aggregation and Processing

aggregated_data = f_data.groupby(['date', 'sku_id', 'fulfilment_location_id']).agg({'order_quantity': 'sum'}).reset_index()
sku_id_map = {val: idx for idx, val in enumerate(aggregated_data['sku_id'].unique())}
fulfilment_location_id_map = {val: idx for idx, val in enumerate(aggregated_data['fulfilment_location_id'].unique())}
aggregated_data['sku_id'] = aggregated_data['sku_id'].map(sku_id_map)
aggregated_data['fulfilment_location_id'] = aggregated_data['fulfilment_location_id'].map(fulfilment_location_id_map)

full_date_range = pd.date_range(start=aggregated_data['date'].min(), end=aggregated_data['date'].max(), freq='W-MON')
product_location_pairs = aggregated_data[['sku_id', 'fulfilment_location_id']].drop_duplicates()
expanded_data = pd.MultiIndex.from_product([full_date_range, product_location_pairs['sku_id'], product_location_pairs['fulfilment_location_id']], names=['date', 'sku_id', 'fulfilment_location_id']).to_frame(index=False)
aggregated_data = expanded_data.merge(aggregated_data, on=['date', 'sku_id', 'fulfilment_location_id'], how='left').fillna({'order_quantity': 0})

def week_of_month(dt):
    first_day = dt.replace(day=1) 
    dom = dt.day 
    adjusted_dom = dom + first_day.weekday()
    return int(np.ceil(adjusted_dom / 7.0)) 

#%%
# Random Forest Model
# MAE ~ 0 for within dates of training range. RF works better than XGBoost etc here. Cannot extrapolate. 

#Create additional time-based features
rf_data = aggregated_data.copy()
rf_data['month'] = rf_data['date'].dt.month
rf_data['week'] = rf_data['date'].dt.isocalendar().week

X = rf_data.drop(columns=['order_quantity', 'date'])
y = rf_data['order_quantity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
model = RandomForestRegressor(n_estimators=1000, random_state=41)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()

#%%
# Random Forest with Week of Month Feature
# Week of month feature to make it applicable to any week. Came from thinking about paydays. 

rf_wom_data = aggregated_data.copy()
rf_wom_data['week_of_month'] = rf_wom_data['date'].apply(week_of_month)

X = rf_wom_data.drop(columns=['order_quantity', 'date'])
y = rf_wom_data['order_quantity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

model = RandomForestRegressor(n_estimators=800, random_state=44)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()

feature_importance = model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance}).sort_values(by='importance', ascending=False)
print(importance_df)

#%%
# Stacking Models
# Looking at stacking models RF, XGBoost, & LightGBM to improve MAE
# Only marginally better than RF alone 

stacking_data = aggregated_data.copy()
stacking_data['week_of_month'] = stacking_data['date'].apply(week_of_month)

X = stacking_data.drop(columns=['order_quantity', 'date'])
print(X.columns)
y = stacking_data['order_quantity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

rf_model = RandomForestRegressor(n_estimators=500, random_state=44)
xgb_model = XGBRegressor(n_estimators=500, random_state=44)
lgb_model = LGBMRegressor(n_estimators=500, random_state=44)
estimators = [('rf', rf_model), ('xgb', xgb_model), ('lgb', lgb_model)]

stacking_model = StackingRegressor(estimators=estimators, final_estimator=GradientBoostingRegressor(n_estimators=1000, random_state=44))
stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()

#%%
# Classification and Regression Approach
# Data imbalance - too many zeros! 
# NEW APPROACH: Two separate models. One for classification to predict whether the demand will 
#               be zero or non-zero, and another for regression to predict the actual demand for 
#               non-zero cases.
# This two stage approach performed better than alternative models with say a Tweedie objective 

clf_reg_data = aggregated_data.copy()
clf_reg_data['week_of_month'] = clf_reg_data['date'].apply(week_of_month)
X = clf_reg_data.drop(columns=['order_quantity', 'date'])
y = clf_reg_data['order_quantity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=41)
y_train_binary = (y_train > 0).astype(int)
clf = RandomForestClassifier(n_estimators=500, random_state=44)
clf.fit(X_train, y_train_binary)
y_pred_class = clf.predict(X_test)

X_train_nonzero = X_train[y_train > 0]
y_train_nonzero = y_train[y_train > 0]
reg = RandomForestRegressor(n_estimators=500, random_state=44)
reg.fit(X_train_nonzero, y_train_nonzero)

y_pred_nonzero = reg.predict(X_test[y_pred_class > 0])
y_pred = np.zeros_like(y_test)
y_pred[y_pred_class > 0] = y_pred_nonzero
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, color='g', label='Actual')
plt.plot(y_pred, color='r', label='Predicted')
plt.legend()
plt.show()

#%%
#%%