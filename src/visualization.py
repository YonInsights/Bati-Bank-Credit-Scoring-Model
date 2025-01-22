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
def plot_outliers(df, numerical_cols):
    """Plot boxplots to visualize outliers in numerical columns."""
    for col in numerical_cols:
        plt.figure(figsize=(8, 5))
        df.boxplot(column=col)
        plt.title(f"Boxplot of {col}")
        plt.show()

def plot_feature_relationships(df, categorical_cols, numerical_cols):
    """Analyze relationships between categorical and numerical variables."""
    for cat_col in categorical_cols:
        for num_col in numerical_cols:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=df, x=cat_col, y=num_col)
            plt.title(f"{num_col} by {cat_col}")
            plt.show()

def compare_distributions(df, group_col, numerical_cols):
    """Compare distributions between groups."""
    unique_groups = df[group_col].unique()
    for num_col in numerical_cols:
        plt.figure(figsize=(8, 5))
        for group in unique_groups:
            sns.kdeplot(df[df[group_col] == group][num_col], label=f"{group}", shade=True)
        plt.title(f"Distribution of {num_col} by {group_col}")
        plt.xlabel(num_col)
        plt.ylabel("Density")
        plt.legend()
        plt.show()