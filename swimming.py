import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from scipy.stats import randint, uniform

df = pd.read_csv("longitudinal_athlete_data_with_RT_corrected.csv")

df_cleaned = df.loc[:, ~df.columns.str.contains("^Unnamed")]  
df_cleaned = df_cleaned.dropna()  
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

selected_features = ["Age", "Lane", "Country", "Final/Heat", 'RT']
target = "100m CT"

df_model = df.dropna(subset=selected_features + [target])
df_model[target] = pd.to_numeric(df_model[target], errors="coerce")
df_model = df_model.dropna(subset=[target])
X_all = pd.get_dummies(df_model[selected_features])
y_all = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=0)

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
cv = KFold(n_splits=5, shuffle=True, random_state=0)

cv = KFold(n_splits=5, shuffle=True, random_state=0)
results = []

for name, (model, params) in models.items():
    search = RandomizedSearchCV(model, params, n_iter=100, cv=cv, scoring="r2", random_state=0, n_jobs=-1)
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




# IOC to ISO-3 manual mapping
ioc2iso = {
    "RSA": "ZAF", "NED": "NLD", "GER": "DEU", "SUI": "CHE", "TPE": "TWN",
    "HKG": "HKG", "URU": "URY", "PRK": "PRK", "CZE": "CZE", "SVK": "SVK",
    "GRE": "GRC", "SLO": "SVN", "IRI": "IRN"
}

def to_iso3(code):
    if not isinstance(code, str) or code.strip() == "":
        return None
    code = code.strip().upper()
    return ioc2iso.get(code, code)

swim_df = pd.read_csv("longitudinal_athlete_data_with_RT_corrected.csv")
gdp_df = pd.read_csv("gdp_2023_nominal.csv")
swim_df.columns = swim_df.columns.str.strip()
gdp_df.columns = gdp_df.columns.str.strip()
swim_df["100m CT"] = pd.to_numeric(swim_df["100m CT"], errors="coerce")
swim_df = swim_df.dropna(subset=["100m CT"])
multi_athletes = swim_df["Athlete"].value_counts()
swim_df = swim_df[swim_df["Athlete"].isin(multi_athletes[multi_athletes >= 2].index)]
swim_df["iso_alpha"] = swim_df["Country"].apply(to_iso3)
swim_df = swim_df.dropna(subset=["iso_alpha"])
country_mean = swim_df.groupby("iso_alpha", as_index=False)["100m CT"].mean()
country_mean.columns = ["iso_alpha", "mean_time"]
gdp_df = gdp_df.rename(columns={"Country Code": "iso_alpha", "2023": "GDP_nominal"})
merged_df = pd.merge(country_mean, gdp_df, on="iso_alpha", how="inner")
rho, pval = spearmanr(merged_df["GDP_nominal"], merged_df["mean_time"])
print(f"Spearman's rho: {rho:.3f}")
print(f"P-value: {pval:.3e}")



