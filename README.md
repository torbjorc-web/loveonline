OKCupid Date A Scientist Project
This project analyzes OKCupid profile data and builds machine learning models to predict a user’s sex from profile information.

Project Goal
The goal is to explore how profile fields such as age, orientation, religion, pets, income, and status relate to a user’s profile category, then evaluate whether a machine learning model can predict that category.

Files
okcupid_one_file.py — one-file Python script with data loading, cleaning, modeling, and plotting.

profiles.csv — the OKCupid dataset (place this in the same folder as the script).

Generated outputs:

target_distribution.png

age_distribution.png

model_comparison.png

top_features.png

model_results.csv

feature_importance.csv

Requirements
Install these Python packages:

bash
pip install pandas numpy matplotlib seaborn scikit-learn
How to Run
Put profiles.csv and okcupid_one_file.py in the same folder.

Open a terminal in that folder.

Run:

bash
python okcupid_one_file.py
What the Script Does
Loads the dataset.

Checks missing values.

Selects useful profile features.

Creates visualizations.

Trains a logistic regression model.

Trains a random forest model.

Compares model accuracy.

Saves feature importance and chart files.

Notes
The script automatically handles missing numeric and categorical values.

If some columns are not present in your dataset version, the script will skip them safely.

You can edit the target variable or features if you want to try a different prediction task.

Possible Extensions
Add essay text analysis with TF-IDF.

Predict a different target such as orientation or zodiac sign.

Try more models such as SVM or XGBoost.

Create a PowerPoint presentation from the results.
