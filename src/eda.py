import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(input_path: str, output_path: str):
    # Загрузка данных
    df = pd.read_csv(input_path)

    # Основная информация
    print("Dataset Info:")
    print(df.info())

    print("\nStatistics:")
    print(df.describe())

    # Создание нового признака
    df["SepalArea"] = df["SepalLengthCm"] * df["SepalWidthCm"]
    df["PetalArea"] = df["PetalLengthCm"] * df["PetalWidthCm"]

    # Сохранение преобразованного датасета
    processed_path = "data/Iris_processed.csv"
    df.to_csv(processed_path, index=False)
    print(f"Processed dataset saved to {processed_path}")

    # Визуализация данных
    sns.pairplot(df, hue="Species")
    plt.savefig(f"{output_path}/pairplot.png")
    print("Pairplot saved to plots folder.")

    # Корреляция признаков
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.savefig(f"{output_path}/correlation_matrix.png")
    print("Correlation matrix saved to plots folder.")

if __name__ == "__main__":
    perform_eda("data/Iris.csv", "plots")
