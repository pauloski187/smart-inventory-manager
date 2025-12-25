import pandas as pd
import numpy as np
from pathlib import Path

# 1. Load Amazon_2_adjusted.csv
df = pd.read_csv('../../data/processed/Amazon 2 adjusted.csv')

# 2. Brand Distribution (Pareto 80/20)
# Simulate power-law for brands
brands = df['Brand'].value_counts().index.tolist()
brand_weights = [0.25, 0.18, 0.12, 0.08, 0.06] + [0.31/25]*25  # Top 5 brands, rest spread
brand_choices = brands[:5] + brands[5:30]
brand_probs = brand_weights
brand_map = np.random.choice(brand_choices, size=len(df), p=brand_probs)
df['Brand'] = brand_map

# Category distribution: Electronics/Home 40-50%
cat_weights = {'Electronics':0.25, 'Home & Kitchen':0.22, 'Clothing':0.18, 'Books':0.13, 'Toys & Games':0.12, 'Sports & Outdoors':0.10}
df['Category'] = np.random.choice(list(cat_weights.keys()), size=len(df), p=list(cat_weights.values()))

# Product: top 20% = 80% sales
products = df['ProductName'].value_counts().index.tolist()
prod_weights = [0.04]*5 + [0.02]*15 + [0.92/len(products[20:])]*(len(products)-20)
df['ProductName'] = np.random.choice(products, size=len(df), p=prod_weights)

# Customers: 80% have 1-2 orders, 5% have 10+
custs = df['CustomerID'].unique()
cust_weights = [0.8/len(custs)]*int(0.8*len(custs)) + [0.15/len(custs)]*int(0.15*len(custs)) + [0.05/len(custs)]*int(0.05*len(custs))
df['CustomerID'] = np.random.choice(custs, size=len(df), p=np.array(cust_weights)[:len(custs)])

# Cities: major metros 10-20x small towns
cities = df['City'].value_counts().index.tolist()
city_weights = [0.15]*5 + [0.05]*10 + [0.8/len(cities[15:])]*(len(cities)-15)
df['City'] = np.random.choice(cities, size=len(df), p=city_weights)

# 2. ShippingCost (Discrete Tiers)
tiers = [0, 5.99, 6.99, 12.99, 14.99, 24.99, 29.99]
tier_weights = [0.65, 0.10, 0.10, 0.06, 0.04, 0.03, 0.02]
df['ShippingCost'] = np.random.choice(tiers, size=len(df), p=tier_weights)

# 3. Add CurrentStock (Column 21)
stock_probs = [0.65, 0.20, 0.10, 0.05]
stock_ranges = [(50,500), (10,49), (500,2000), (0,9)]
stock_choices = np.random.choice([0,1,2,3], size=len(df), p=stock_probs)
current_stock = [np.random.randint(*stock_ranges[i]) for i in stock_choices]
df['CurrentStock'] = current_stock

# 4. Discounts: 70% = 0%, 20% = 5-15%, 10% = 20%+
discount_probs = [0.7, 0.2, 0.1]
discount_ranges = [(0,0), (0.05,0.15), (0.2,0.4)]
discount_choices = np.random.choice([0,1,2], size=len(df), p=discount_probs)
df['Discount'] = [np.round(np.random.uniform(*discount_ranges[i]),2) for i in discount_choices]

# 5. Order Values: Log-normal for UnitPrice, right-skewed TotalAmount
unit_price = np.random.lognormal(mean=5, sigma=0.5, size=len(df))
df['UnitPrice'] = np.round(unit_price,2)
df['TotalAmount'] = df['Quantity']*df['UnitPrice'] + df['Tax'] + df['ShippingCost'] - df['Discount']

# Save as Amazon_sales.csv
df.to_csv('../data/processed/Amazon_sales.csv', index=False)
print('Amazon_sales.csv created.')
