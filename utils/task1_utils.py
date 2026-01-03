# utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_class_distribution(class_counts, figsize=(8, 8)):
    """
    Plot a simple histogram (bar chart) for class/sample counts.

    Parameters
    ----------
    class_counts : pd.Series
        Pandas value_counts() result (index=labels, values=counts)
    figsize : tuple, optional
        Figure size (width, height), by default (8, 8)
    """
    class_counts = class_counts.sort_values(ascending=True)

    plt.figure(figsize=figsize)
    ax = class_counts.plot(kind='barh', color='steelblue')

    ax.set_title("Sample counts per medical specialty", fontsize=18)
    ax.set_xlabel("Number of Samples", fontsize=12)
    ax.set_ylabel("Medical Specialty", fontsize=12)

    plt.yticks(rotation=0, fontsize=11)

    # Add value labels
    for i, value in enumerate(class_counts.values):
        ax.text(
            value,
            i,
            f"{value}",
            va='center',
            ha='left',
            fontsize=10
        )

    plt.tight_layout()
    plt.show()


def top_terms_per_cluster(X, labels, vectorizer, n_terms=10):
    feature_names = vectorizer.get_feature_names_out()
    clusters = {}
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        means = np.asarray(X[idx].mean(axis=0)).ravel()
        top_idx = means.argsort()[::-1][:n_terms]
        clusters[c] = feature_names[top_idx]
    return clusters

# dual metric plot
def plot_metrics(results_df, metric):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.plot(results_df["K"], results_df[metric], marker="o", label=metric)  
    plt.xlabel("K (#clusters)")
    plt.ylabel("Score")
    plt.title("Clustering metrics vs K")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_metrics_bar(results_df,title="Clustering Metrics",figsize=(22, 10)):
    metrics = ["silhouette", "homogeneity", "completeness", "v_measure", "ARI"]

    pipelines = results_df["config"].tolist()

    data = results_df[metrics].values.T  # 5 x num_pipeline

    fig, axes = plt.subplots(1, 5, figsize=figsize, sharey=False)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = data[i]

        ax.bar(np.arange(len(pipelines)), values, color="royalblue")

        ax.set_title(metric, fontsize=10)
        ax.set_xticks(np.arange(len(pipelines)))
        ax.set_xticklabels(pipelines, rotation=70)

        for j, v in enumerate(values):
            ax.text(j, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()




def plot_two_metrics(results_df, metric_left, metric_right):
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7, 4))

    ax1.plot(results_df["K"], results_df[metric_left], marker="o", color="tab:blue", label=metric_left)
    ax1.set_xlabel("Number of clusters (K)")
    ax1.set_ylabel(metric_left, color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(results_df["K"], results_df[metric_right], marker="s", color="tab:orange", label=metric_right)
    ax2.set_ylabel(metric_right, color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title(f"{metric_left} vs {metric_right}")
    fig.tight_layout()
    plt.show()



# cluster vs label heatmap
def plot_crosstab(df,figsize=(12,8)):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # contingency table
    table = pd.crosstab(df['cluster'], df['label'])
    
    # sort columns (optional, easier interpretation)
    table = table[table.sum().sort_values(ascending=False).index]

    plt.figure(figsize=figsize)
    sns.heatmap(table.T, cmap="Blues", cbar_kws=dict(shrink=0.6), linewidths=0.5, linecolor ='#EEEEEE' )

    plt.title("Cluster vs Label Heatmap", fontsize=10)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.show()


# cluster size bar plot
def plot_cluster_sizes(df):
    import matplotlib.pyplot as plt

    counts = df["cluster"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(10, 4))
    ax = counts.plot(kind="bar", color="steelblue")
    plt.title("Cluster Size Distribution", fontsize=30)
    plt.xlabel("Cluster ID (sorted by size)")
    plt.ylabel("# Documents")

    # add percentage labels
    total = len(df)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, f"{v/total*100:.1f}%", ha="center", fontsize=8)

    plt.tight_layout()
    plt.show()


# t-SNE of clusters
def plot_tsne(X, labels):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np

    # convert sparse --> dense only when needed
    if hasattr(X, "toarray"):
        X = X.toarray()

    X_2d = TSNE(
        random_state=42,
        perplexity=40,
        init="pca",
        learning_rate="auto"
    ).fit_transform(X)

    plt.figure(figsize=(6,5))
    plt.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap="tab20", s=5)
    plt.title("t-SNE of clusters")
    plt.show()
    
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics_bar_rotated(results_df):
    metrics = ["silhouette", "homogeneity", "completeness", "v_measure", "ARI"]
    pipelines = results_df["config"].tolist()

    # extract metric values preserving ordering
    data = results_df[metrics].values.T  # shape now 5 x num_pipeline

    # Define colors for each pipeline (grouped in pairs with lighter/darker shades)
    colors = [
        '#D3D3D3',  # Pipeline 0 - light gray (TFIDF_NONE)
        '#A9A9A9',  # Pipeline 1 - darker gray (TFIDF_PROCESSED)
        '#B0C4DE',  # Pipeline 2 - light steel blue (TFIDF_NONE_WEIGHTED)
        '#6495ED',  # Pipeline 3 - darker steel blue (TFIDF_PROCESSED_WEIGHTED)
        '#FF9999',  # Pipeline 4 - light red (SEMANTIC_EMBEDDING_NONE)
        '#FF0000',  # Pipeline 5 - red (SEMANTIC_EMBEDDING_PROCESSED)
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = data[i]

        # horizontal bar plot with grouped colors
        y_pos = np.arange(len(pipelines))
        bars = ax.barh(y_pos, values, color=[colors[j] for j in range(len(pipelines))])

        ax.set_title(metric, fontsize=20)
        
        # Only show y-labels on the leftmost plot
        if i == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(pipelines, fontsize=20)
        
        # Set x-axis tick label fontsize
        ax.tick_params(axis='x', labelsize=20)
        
        # Add grid to all subplots
        ax.grid(axis='x', color='lightgray', linestyle='-', linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)  # Put grid behind bars
        
        # For silhouette (first plot), set custom x-axis limit and ticks
        if i == 0:
            max_val = max(values)
            xlim_max = max_val * 1.1  # 1.1x the max value
            ax.set_xlim(0, xlim_max)
            # Set custom x-ticks
            ax.set_xticks([0.0, 0.05, 0.10])
       
        ax.invert_yaxis()  # highest pipeline on top

        # # annotate value to the right of bar
        # for j, v in enumerate(values):
        #     ax.text(v, j, f" {v:.3f}", ha="left", va="center", fontsize=20)

    # Only label y-axis on the leftmost subplot
    axes[0].set_ylabel("PIPELINE", fontsize=20)
    
    fig.suptitle("Comparison of Pipelines across Metrics", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

