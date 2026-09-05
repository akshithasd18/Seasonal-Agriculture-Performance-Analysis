# Seasonal Agriculture Performance Analysis
# Major Project - Data Analytics
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------
DATA_FILE = "seasonal_agriculture_performance_dataset.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_FILE)
print("Dataset Shape:", df.shape)
print("\nFirst 5 Records:")
print(df.head())
print("\nDataset Information:")
print(df.info())
# --------------------------------------------------
# 2. DATA CLEANING
# --------------------------------------------------
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Records:", df.duplicated().sum())
# Remove duplicate records
df = df.drop_duplicates()
# Fill missing numerical values using median
numeric_columns = df.select_dtypes(include=np.number).columns
for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())
# Fill missing categorical values using mode
categorical_columns = df.select_dtypes(include="object").columns
for column in categorical_columns:
    if df[column].isnull().any():
        df[column] = df[column].fillna(df[column].mode()[0])
print("\nCleaned Dataset Shape:", df.shape)
# --------------------------------------------------
# 3. DESCRIPTIVE STATISTICS
# --------------------------------------------------
print("\nDescriptive Statistics:")
print(df.describe())
# --------------------------------------------------
# 4. SEASONAL YIELD ANALYSIS
# --------------------------------------------------
season_yield = df.groupby("Season")["Yield_Tonnes_Ha"].mean()
print("\nAverage Yield by Season:")
print(season_yield)
plt.figure(figsize=(10, 6))
season_yield.plot(kind="bar")
plt.title("Average Agricultural Yield by Season")
plt.xlabel("Season")
plt.ylabel("Average Yield (Tonnes/Ha)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig( f"{OUTPUT_DIR}/seasonal_yield.png",dpi=300)
plt.show()
# --------------------------------------------------
# 5. SEASONAL PROFIT ANALYSIS
# --------------------------------------------------
season_profit = df.groupby("Season")["Profit_INR"].mean()
print("\nAverage Profit by Season:")
print(season_profit)
plt.figure(figsize=(10, 6))
season_profit.plot(kind="bar")
plt.title("Average Profit by Season")
plt.xlabel("Season")
plt.ylabel("Average Profit (INR)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/seasonal_profit.png",dpi=300)
plt.show()
# --------------------------------------------------
# 6. CROP-WISE YIELD ANALYSIS
# --------------------------------------------------
crop_yield = (
    df.groupby("Crop")["Yield_Tonnes_Ha"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Yield by Crop:")
print(crop_yield)

plt.figure(figsize=(11, 6))
crop_yield.plot(kind="bar")
plt.title("Average Yield by Crop")
plt.xlabel("Crop")
plt.ylabel("Average Yield (Tonnes/Ha)")
plt.xticks(rotation=35)
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/crop_yield.png",
    dpi=300
)
plt.show()

# --------------------------------------------------
# 7. STATE-WISE YIELD ANALYSIS
# --------------------------------------------------

state_yield = (
    df.groupby("State")["Yield_Tonnes_Ha"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Yield by State:")
print(state_yield)

plt.figure(figsize=(11, 6))
state_yield.plot(kind="bar")
plt.title("Average Agricultural Yield by State")
plt.xlabel("State")
plt.ylabel("Average Yield (Tonnes/Ha)")
plt.xticks(rotation=35)
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/state_yield.png",
    dpi=300
)
plt.show()

# --------------------------------------------------
# 8. IRRIGATION METHOD ANALYSIS
# --------------------------------------------------

irrigation_analysis = df.groupby("Irrigation_Method").agg(
    Average_Yield=("Yield_Tonnes_Ha", "mean"),
    Average_Water_Efficiency=(
        "Water_Efficiency_t_per_1000m3",
        "mean"
    ),
    Average_Profit=("Profit_INR", "mean")
)

print("\nIrrigation Method Analysis:")
print(irrigation_analysis)

irrigation_analysis[
    ["Average_Yield", "Average_Water_Efficiency"]
].plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Irrigation Method Comparison")
plt.xlabel("Irrigation Method")
plt.ylabel("Average Value")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/irrigation_comparison.png",
    dpi=300
)
plt.show()

# --------------------------------------------------
# 9. ENVIRONMENTAL FACTORS ANALYSIS
# --------------------------------------------------

environmental_columns = [
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Humidity_pct",
    "Sunlight_Hours_Day",
    "Soil_pH",
    "Soil_Moisture_pct"
]

environmental_yield = df[
    environmental_columns + ["Yield_Tonnes_Ha"]
].corr()["Yield_Tonnes_Ha"].sort_values(
    ascending=False
)

print("\nCorrelation of Environmental Factors with Yield:")
print(environmental_yield)

# --------------------------------------------------
# 10. RESOURCE USAGE ANALYSIS
# --------------------------------------------------

resource_columns = [
    "Nitrogen_kg_ha",
    "Phosphorus_kg_ha",
    "Potassium_kg_ha",
    "Fertilizer_kg_ha",
    "Pesticide_Litre_ha",
    "Water_Used_m3"
]

resource_yield = df[
    resource_columns + ["Yield_Tonnes_Ha"]
].corr()["Yield_Tonnes_Ha"].sort_values(
    ascending=False
)

print("\nCorrelation of Resource Factors with Yield:")
print(resource_yield)

# --------------------------------------------------
# 11. CORRELATION HEATMAP
# --------------------------------------------------

correlation_columns = [
    "Rainfall_mm",
    "Avg_Temperature_C",
    "Humidity_pct",
    "Soil_pH",
    "Soil_Moisture_pct",
    "Fertilizer_kg_ha",
    "Water_Used_m3",
    "Yield_Tonnes_Ha",
    "Market_Price_INR_Tonne",
    "Total_Cost_INR",
    "Revenue_INR",
    "Profit_INR",
    "Disease_Pest_Risk_pct"
]

correlation_matrix = df[correlation_columns].corr()

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap of Agricultural Factors")
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/correlation_heatmap.png",
    dpi=300
)
plt.show()

# --------------------------------------------------
# 12. SEASONAL PRODUCTION ANALYSIS
# --------------------------------------------------

season_production = (
    df.groupby("Season")["Production_Tonnes"]
    .mean()
)

print("\nAverage Production by Season:")
print(season_production)

plt.figure(figsize=(10, 6))
season_production.plot(kind="bar")

plt.title("Average Production by Season")
plt.xlabel("Season")
plt.ylabel("Production (Tonnes)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/seasonal_production.png",
    dpi=300
)
plt.show()

# --------------------------------------------------
# 13. DISEASE / PEST RISK ANALYSIS
# --------------------------------------------------

season_risk = (
    df.groupby("Season")["Disease_Pest_Risk_pct"]
    .mean()
)

print("\nAverage Disease/Pest Risk by Season:")
print(season_risk)

plt.figure(figsize=(10, 6))
season_risk.plot(kind="bar")

plt.title("Disease and Pest Risk by Season")
plt.xlabel("Season")
plt.ylabel("Risk (%)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/seasonal_risk.png",
    dpi=300
)
plt.show()

# --------------------------------------------------
# 14. ECONOMIC PERFORMANCE
# --------------------------------------------------

economic_analysis = df.groupby("Season").agg(
    Average_Cost=("Total_Cost_INR", "mean"),
    Average_Revenue=("Revenue_INR", "mean"),
    Average_Profit=("Profit_INR", "mean")
)

print("\nEconomic Performance by Season:")
print(economic_analysis)

# --------------------------------------------------
# 15. FINAL SUMMARY
# --------------------------------------------------

best_season_yield = season_yield.idxmax()
best_crop = crop_yield.idxmax()
best_state = state_yield.idxmax()
best_irrigation = irrigation_analysis[
    "Average_Yield"
].idxmax()
print("\n======================================")
print("FINAL PROJECT INSIGHTS")
print("======================================")
print("Best Season based on Yield:",best_season_yield)
print("Highest Yielding Crop:",best_crop)
print("Highest Yielding State:",best_state)
print("Best Irrigation Method based on Yield:", best_irrigation)
print("\nAnalysis completed successfully!")
print("All graphs are saved inside the 'outputs' folder.")
