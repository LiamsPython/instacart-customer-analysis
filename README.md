# Instacart Customer Analysis

Insightful data-driven analysis of customer purchasing patterns using Python & visualisations.

# Project Overview

This project examines over 3 million anonymized Instacart grocery orders to uncover:
Peak shopping windows (weekday/hour trends)
Product popularity & loyalty
Market basket relationships & association rules
Customer segmentation and reorder prediction
Using Python (pandas, seaborn), Jupyter, and clear visual storytelling 🎯.

# Motivation

With vast grocery data, the goal is to transform raw numbers into actionable insights:
Help marketers time promotions
Aid inventory teams in stocking high-demand items
Support recommendation systems for improved customer experience

# Data & Preparation

Data sources: orders, products, departments, aisles, and customer data.
Dataset obtained from Kaggle: https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset

Processing steps: Type corrections, deduplication, and null handling
Memory optimization via efficient data formats
Sampling techniques for performance

# Analysis Workflow

Exploratory Data Analysis (EDA)
Identify order patterns by day and hour
Analyze order intervals, basket size, and repeat behavior
Market Basket Analysis
Employ Apriori algorithm to reveal product co-purchasing patterns
Customer Segmentation
Cluster analysis to define shopper types (e.g., fresh produce lovers vs. bulk buyers)
Reorder Prediction
Feature engineering (user, product, aisle, order features)
Baseline model for predicting if an item will be reordered

# Key Findings

Peak hours: 10–11 am & 2–3 pm; busiest days: Sundays & Mondays
Repeat purchase behavior: ~11-day average reordering interval; 58–60% of products are repeat buys
Top items: Bananas, Organic Berries, Avocados consistently rank at the top
Association rules: Frequent combos include (Bananas ↔ Organic Raspberries), (Limes ↔ Lemons)
Segments identified: Distinct shopper types such as organic-lovers, beverage-focused, and occasional shoppers

# Usage

# Clone the repository
git clone https://github.com/LiamsPython/instacart-customer-analysis.git
cd instacart-customer-analysis/

# (Optional) Set up virtual environment
conda env create -f environment.yml   # Or use pip

# Run the notebook
jupyter notebook Instacart_Customer_Analysis.ipynb

Note: For large dataset handling, ensure the required .csv.gz or efficient formats are in place. See "Data & Preparation" for tips.

Results & Visualizations

Check the Jupyter notebook for detailed visuals:
Time-series heatmaps (orders by hour/day)
Histograms of basket sizes & reorder rates
Network graphs from basket analysis
Customer cluster profiles & reorder model performance charts

Future Directions

Enhance reorder prediction using machine learning (e.g., XGBoost or CatBoost)
Build a live dashboard (Plotly/Dash or Tableau)
Integrate demographic features for deeper customer insights
Automate data pipeline for refreshable visuals

About the Author:

Liam – Python enthusiast & data storyteller.

📨 Reach out: sheppy144@gmail.com

Explore more at my GitHub portfolio.

Enjoyed this analysis?

Star ⭐ the repo & follow for more data science deep-dives!
