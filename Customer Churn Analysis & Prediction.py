import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():

    df = pd.read_excel("Telco_customer_churn.xlsx")

    print(df.head())
    print(df.info())

    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    df['Total Charges'] = df['Total Charges'].fillna(df['Total Charges'].median())

    drop_cols = ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
                 'Lat Long', 'Latitude', 'Longitude', 'Churn Score', 'CLTV', 'Churn Reason']

    df = df.drop(columns=drop_cols, errors='ignore')

    print("\nChurn Distribution:")
    print(df['Churn Label'].value_counts())

    churn_rate = (df['Churn Label'] == 'Yes').mean() * 100
    print(f"\nOverall Churn Rate: {churn_rate:.2f}%")

    print("\nChurn by Contract (%):")
    print(pd.crosstab(df['Contract'], df['Churn Label'], normalize='index') * 100)

    print("\nAverage Monthly Charges:")
    print(df.groupby('Churn Label')['Monthly Charges'].mean())

    print("\nAverage Tenure:")
    print(df.groupby('Churn Label')['Tenure Months'].mean())

    plt.figure()
    sns.countplot(x='Churn Label', data=df)
    plt.title("Churn Distribution")
    plt.show()

    plt.figure()
    sns.boxplot(x='Churn Label', y='Monthly Charges', data=df)
    plt.title("Monthly Charges vs Churn")
    plt.show()

    plt.figure()
    sns.boxplot(x='Churn Label', y='Tenure Months', data=df)
    plt.title("Tenure vs Churn")
    plt.show()

    plt.figure()
    sns.countplot(x='Contract', hue='Churn Label', data=df)
    plt.title("Contract Type vs Churn")
    plt.xticks(rotation=20)
    plt.show()

    plt.figure()
    sns.countplot(x='Internet Service', hue='Churn Label', data=df)
    plt.title("Internet Service vs Churn")
    plt.show()

    plt.figure()
    sns.countplot(x='Payment Method', hue='Churn Label', data=df)
    plt.title("Payment Method vs Churn")
    plt.xticks(rotation=30)
    plt.show()

    plt.figure(figsize=(10,6))
    sns.histplot(data=df, x='Monthly Charges', hue='Churn Label', kde=True)
    plt.title("Monthly Charges Distribution")
    plt.show()

    df_encoded = df.copy()
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            df_encoded[col] = le.fit_transform(df_encoded[col])

    plt.figure(figsize=(12,8))
    sns.heatmap(df_encoded.corr(), cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

if __name__ == "__main__":
    main()