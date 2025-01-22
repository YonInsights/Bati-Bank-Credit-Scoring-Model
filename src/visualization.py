import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(df, column, bins=30):
    """Plot a histogram for a specific column."""
    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], bins=bins, kde=True, color='blue')
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

def plot_correlation_matrix(df):
    """Plot a correlation matrix."""
    plt.figure(figsize=(10, 8))

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
  
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

def plot_bar_chart(df, column):
    """Plot a bar chart for a categorical column."""
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=column, palette="viridis")
    plt.title(f"Bar Chart of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()
