# utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_class_distribution(class_counts):
    """
    Plot a simple histogram (bar chart) for class/sample counts.
    
    Parameters
    ----------
    class_counts : pd.Series
        Pandas value_counts() result (index=labels, values=counts)
    """
    class_counts = class_counts.sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    class_counts.plot(kind='bar', color='steelblue')
    plt.title("Sample counts per medical specialty")
    plt.xlabel("Medical Specialty")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=90)
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
def plot_crosstab(df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # contingency table
    table = pd.crosstab(df['cluster'], df['label'])
    
    # sort columns (optional, easier interpretation)
    table = table[table.sum().sort_values(ascending=False).index]

    plt.figure(figsize=(18, 10))
    sns.heatmap(table.T, cmap="Blues", cbar_kws=dict(shrink=0.6), linewidths=0.5, linecolor ='#EEEEEE' )

    plt.title("Cluster vs Label Heatmap", fontsize=14)
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
    plt.title("Cluster Size Distribution")
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

def plot_metrics_bar(results_df):
    metrics = ["silhouette", "homogeneity", "completeness", "v_measure", "ARI"]
    pipelines = results_df["pipeline"].tolist()

    # extract metric values preserving ordering
    data = results_df[metrics].values.T  # shape now 5 x num_pipeline

    fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=False)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = data[i]

        # one column bar plot
        ax.bar(np.arange(len(pipelines)), values, color="royalblue")

        ax.set_title(metric, fontsize=12)
        ax.set_xticks(np.arange(len(pipelines)))
        ax.set_xticklabels(pipelines, rotation=70)

        # annotate value above bar
        for j, v in enumerate(values):
            ax.text(j, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Comparison of Pipelines across Metrics", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

