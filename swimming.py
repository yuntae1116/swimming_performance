import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from scipy.stats import randint, uniform
from sklearn.metrics import mean_absolute_error
import seaborn as sns





df = pd.read_csv("longitudinal_athlete_data_with_RT_corrected.csv")




df_cleaned = df.loc[:, ~df.columns.str.contains("^Unnamed")]  
df_cleaned = df_cleaned.dropna()  




df_cleaned.head(3)




df_cleaned.describe()




df_cleaned = df_cleaned[[
    "Athlete", "Year", "Event", "Event Type", "Final/Heat",
    "Country", "Age", "gender", "Rank", "Lane", "100m", "100m CT",'RT'
]]
df_cleaned = df_cleaned.sort_values(by=["Athlete", "Year"]).reset_index(drop=True)
df_cleaned["100m CT"] = pd.to_numeric(df_cleaned["100m CT"], errors="coerce")

athlete_years = df_cleaned.groupby("Athlete")["Year"].nunique().reset_index()
athlete_years.columns = ["Athlete", "Num_Years"]
df_with_counts = df_cleaned.merge(athlete_years, on="Athlete")
df_long = df_with_counts[df_with_counts["Num_Years"] >= 2]




df_long.describe()




df_age_perf = df_long.dropna(subset=["Age", "100m CT"])
df_age_perf["100m CT"] = pd.to_numeric(df_age_perf["100m CT"], errors="coerce")
df_age_perf = df_age_perf.dropna(subset=["100m CT"])

X_age = df_age_perf["Age"].values.reshape(-1, 1)
y_time = df_age_perf["100m CT"].values
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_age)
model_poly = LinearRegression().fit(X_poly, y_time)




beta_0 = model_poly.intercept_
beta_1, beta_2 = model_poly.coef_[1], model_poly.coef_[2]
peak_age = -beta_1 / (2 * beta_2)




print(beta_0, beta_1, beta_2)




age_range = np.linspace(df_age_perf["Age"].min(), df_age_perf["Age"].max(), 300).reshape(-1, 1)
age_range_poly = poly.transform(age_range)
predicted_times = model_poly.predict(age_range_poly)




print("===== Aging Curve Analysis (Polynomial Regression) =====")
print(f"Peak Age: {peak_age:.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_time, model_poly.predict(X_poly))):.3f}")
print(f"R^2: {r2_score(y_time, model_poly.predict(X_poly)):.3f}")




plt.figure(figsize=(10, 6))

plt.scatter(
    df_age_perf["Age"],
    df_age_perf["100m CT"],
    alpha=0.5,
    linewidth=0.3,
    label="Individual Records"
)

plt.plot(
    age_range,
    predicted_times,
    linewidth=2.5,
    label="Aging Curve (Polynomial Fit)"
)

plt.xlabel("Age", fontsize=12)
plt.ylabel("Record", fontsize=12)
# plt.title("Aging Curve for Male 100m Freestyle", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
# plt.savefig("aging_curve.png", dpi=300)
plt.show()




selected_features = ["Age", "Lane", "Country", "Final/Heat", 'RT']
target = "100m CT"

df_model = df.dropna(subset=selected_features + [target])
df_model[target] = pd.to_numeric(df_model[target], errors="coerce")
df_model = df_model.dropna(subset=[target])
X_all = pd.get_dummies(df_model[selected_features])
y_all = df_model[target]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)




# Model definitions
models = {
    "Random Forest": (RandomForestRegressor(), {
        "n_estimators": randint(100, 300),
        "max_depth": randint(5, 20),
        "min_samples_split": randint(2, 10)
    }),
    "Gradient Boosting": (GradientBoostingRegressor(), {
        "n_estimators": randint(100, 300),
        "learning_rate": uniform(0.01, 0.2),
        "max_depth": randint(3, 10)
    }),
    "MLP Regressor": (MLPRegressor(max_iter=1000), {
        "hidden_layer_sizes": [(100,), (100, 50), (128, 64)],
        "activation": ['relu', 'tanh'],
        "alpha": uniform(0.0001, 0.01),
        "learning_rate_init": uniform(0.001, 0.01),
        "batch_size": [32, 64, 128]
    }),
        "hidden_layer_sizes": [(100,), (100, 50)],
        "alpha": uniform(0.0001, 0.01),
        "learning_rate_init": uniform(0.001, 0.01)
    })
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)




cv = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

from sklearn.metrics import mean_absolute_error

for name, (model, params) in models.items():
    search = RandomizedSearchCV(model, params, n_iter=100, cv=cv, scoring="r2", random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    y_pred_test = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)

    results.append((name, mae, rmse, r2))




print("\n===== Model Performance on Test Set (CV-tuned) =====")
for name, mae, rmse, r2 in results:
    print(f"{name}: MAE= {mae: .3f}, RMSE = {rmse:.3f}, R² = {r2:.3f}")




gbm_model, gbm_params = models["Gradient Boosting"]
search = RandomizedSearchCV(gbm_model, gbm_params, n_iter=100, cv=cv,
                            scoring="r2", random_state=42, n_jobs=-1)
search.fit(X_train, y_train)
best_gbm_model = search.best_estimator_

importances = best_gbm_model.feature_importances_




feature_names = X_train.columns
importances    = best_gbm_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# ──────────────────────────────────────────────
# ① Country_*  → "Country"
# ② Final/Heat_* → "Competition Type"
def _merge_name(col):
    if col.startswith("Country_"):
        return "Country"
    elif col.startswith("Final/Heat_"):
        return "Competition Type"
    else:
        return col

importance_df["Feature"] = importance_df["Feature"].apply(_merge_name)

importance_df = (importance_df
                 .groupby("Feature", as_index=False)["Importance"]
                 .sum()
                 .sort_values("Importance", ascending=False))

# ──────────────────────────────────────────────
plt.figure(figsize=(8, 6))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
# plt.title("Feature Importances from GBM")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig('feature_importance.png', dpi = 300)
plt.show()




y_train_pred = best_gbm_model.predict(X_train)
y_test_pred = best_gbm_model.predict(X_test)

plt.figure(figsize=(7, 6))
sns.scatterplot(x=y_train, y=y_train_pred, label='Train', color='blue', alpha=0.5)
sns.scatterplot(x=y_test, y=y_test_pred, label='Test', color='orange', alpha=0.7)

min_val = min(y_train.min(), y_test.min(), y_train_pred.min(), y_test_pred.min())
max_val = max(y_train.max(), y_test.max(), y_train_pred.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y = ŷ)')

plt.xlabel("Actual 100m Time (seconds)")
plt.ylabel("Predicted 100m Time (seconds)")
# plt.title("Comparison of Actual and Predicted Values")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('prediction_error_swimming.png', dpi = 300)
plt.show()




country_stats = df_long.groupby('Country').agg(
    count=('Athlete', 'count'),
    unique_athletes=('Athlete', 'nunique'),
    mean_time=('100m', 'mean'),
    std_time=('100m', 'std'),
    mean_RT=('RT', 'mean'),
    mean_age=('Age', 'mean')
).reset_index()




def to_iso3(code):
    try:
        return pycountry.countries.lookup(code).alpha_3
    except:
        return None

country_stats['iso_alpha'] = country_stats['Country'].str.upper().apply(to_iso3)
country_stats = country_stats.dropna(subset=['iso_alpha'])




import pandas as pd
import pycountry
import plotly.express as px
from typing import Optional

ioc2iso = {
    "RSA": "ZAF", "NED": "NLD", "GER": "DEU", "SUI": "CHE", "TPE": "TWN",
    "HKG": "HKG", "URU": "URY", "PRK": "PRK", "CZE": "CZE", "SVK": "SVK",
    "GRE": "GRC", "SLO": "SVN", "IRI": "IRN"
}
all_iso3 = {c.alpha_3 for c in pycountry.countries}

def to_iso3(code: str) -> Optional[str]:
    if not isinstance(code, str) or code.strip() == "":
        return None
    code = code.strip().upper()
    code = ioc2iso.get(code, code)
    return code if code in all_iso3 else None

swim = pd.read_csv("longitudinal_athlete_data_with_RT_corrected.csv")
swim.columns = swim.columns.str.strip()

swim["100m CT"] = pd.to_numeric(swim["100m CT"], errors="coerce")
swim = swim.dropna(subset=["100m CT"])

multi_mask = swim["Athlete"].value_counts()
swim = swim[swim["Athlete"].isin(multi_mask[multi_mask >= 2].index)]

swim["iso_alpha"] = swim["Country"].apply(to_iso3)
swim = swim.dropna(subset=["iso_alpha"])

country_stats = swim.groupby("iso_alpha", as_index=False).agg(
    mean_time=("100m CT", "mean"),
    Country=("Country", "first")
)

world = pd.DataFrame({"iso_alpha": sorted(all_iso3)})
merged = world.merge(
    country_stats[["iso_alpha", "Country", "mean_time"]],
    on="iso_alpha", how="left"
)

fig = px.choropleth(
    merged,
    locations="iso_alpha",
    color="mean_time",
    hover_name="Country",
    color_continuous_scale=list(reversed(px.colors.sequential.Reds)),
    range_color=(48, 50),
    labels={"mean_time": "Avg 100 m Time (s)"}
)
fig.update_layout(width=1000, height=600)
fig.update_geos(showcoastlines=True, showcountries=True)
fig.show()




import matplotlib.cm as cm

df = pd.read_csv("longitudinal_athlete_data_with_RT_corrected.csv")
df.columns = df.columns.str.strip()

df['100m_CT'] = pd.to_numeric(df['100m CT'], errors='coerce')
df = df.dropna(subset=['100m_CT'])
df['100m_s'] = df['100m_CT'] / 100

multi_athletes = df['Athlete'].value_counts()
df_long = df[df['Athlete'].isin(multi_athletes[multi_athletes >= 2].index)]

ioc2iso = {"RSA": "ZAF"}
all_iso3 = {c.alpha_3 for c in pycountry.countries}
def to_iso3(code):
    if not isinstance(code, str): return None
    code = code.strip().upper()
    return ioc2iso.get(code, code) if code in all_iso3 or code in ioc2iso else None

df_long['iso_alpha'] = df_long['Country'].apply(to_iso3)

threshold = df_long["100m_s"].quantile(0.10)
top_performers = df_long[df_long["100m_s"] <= threshold]
top_record_counts = top_performers["iso_alpha"].value_counts()
top10_counts = top_record_counts.head(10)
other_count = top_record_counts.iloc[10:].sum()

pie_data = top10_counts.copy()
if other_count > 0:
    pie_data["Other"] = other_count

pie_labels = ["RSA" if iso == "ZAF" else iso for iso in pie_data.index]

colors = list(cm.tab20.colors[:len(pie_data)-1]) + ['#999999']
fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.pie(
    pie_data.values,
    labels=pie_labels,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    textprops={'fontsize': 9}
)
ax1.axis('equal')
plt.tight_layout()
plt.savefig("pie_chart_swimming.png", dpi=300)
plt.show()




ordered_iso = [iso for iso in pie_data.index if iso != "Other"]
swim_rank_by_count = {iso: rank + 1 for rank, iso in enumerate(ordered_iso)}

top10_df = df_long[df_long["iso_alpha"].isin(ordered_iso)]

summary_df = top10_df.groupby("iso_alpha", as_index=False).agg(
    mean_time=("100m_s", "mean")
)
summary_df["Swim_Rank"] = summary_df["iso_alpha"].map(swim_rank_by_count)

gdp = pd.read_csv("gdp_2023_nominal.csv")
gdp.columns = gdp.columns.str.strip()
gdp = gdp[gdp["Country Code"].isin(all_iso3)].dropna(subset=["2023"])
gdp["GDP Rank"] = gdp["2023"].rank(method="min", ascending=False).astype(int)

summary_df["GDP Rank"] = summary_df["iso_alpha"].map(
    gdp.set_index("Country Code")["GDP Rank"]
)
summary_df = summary_df.set_index("iso_alpha").loc[ordered_iso].reset_index()
rankrank_df = summary_df.rename(columns={"iso_alpha": "ISO3"})

x_vals = rankrank_df["GDP Rank"].values
y_vals = rankrank_df["Swim_Rank"].values
rank_labels = ["RSA" if iso == "ZAF" else iso for iso in rankrank_df["ISO3"]]

slope, intercept, r_val, p_val, _ = linregress(x_vals, y_vals)
x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
y_line = slope * x_line + intercept

fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.scatter(
    x_vals, y_vals,
    s=100, facecolor="#4A90E2", edgecolor="black", linewidth=0.8
)
for i in range(len(rank_labels)):
    ax2.text(
        x_vals[i], y_vals[i] - 0.25,
        rank_labels[i], fontsize=9, ha="center", va="top", color="black"
    )

ax2.plot(x_line, y_line, linestyle="--", color="gray", linewidth=1.5)
ax2.invert_xaxis()
ax2.invert_yaxis()
ax2.set_xlabel("GDP Rank", fontsize=11)
ax2.set_ylabel("Swimming Rank\n(based on Top-10% Record Count)", fontsize=10)
ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

plt.tight_layout()
plt.savefig("rank_rank_plot_swimming.png", dpi=300)
plt.show()


# In[ ]:





# ### Note on Visualizations
# This notebook focuses on machine learning-based performance prediction and aging analysis.
# 
# - Global map and pie chart visualizations (Figures 4 and 5 in the manuscript) are excluded here for simplicity.
# - These figures serve illustrative purposes in the publication and do not affect the core model results.
# - Athlete-level data is based on publicly available sources and may be updated periodically.
# 

# In[ ]:


# Dummy placeholder for global map and pie chart (Figures 4 and 5)
# Actual visualization code omitted in GitHub version for simplicity
print('Visualizations for world map and pie chart are omitted. Refer to the manuscript for figures.')

