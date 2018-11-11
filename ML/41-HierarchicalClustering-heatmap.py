import seaborn as sns
iris = sns.load_dataset('iris')
species = iris.pop("species")
print(iris.shape)
print(type(iris))

# Adding color code for each flower species
lut = dict(zip(species.unique(), "rbg"))
row_colors = species.map(lut)
print(lut)
# euclidean distance and average method for clustering
g = sns.clustermap(iris, row_cluster = True, col_cluster = False,
                   method = 'average', metric= 'euclidean', row_colors  = row_colors)
g.savefig('Iris-heatmap-averageMethod.png')

# euclidean distance and minimum cluster distance method for clustering
g = sns.clustermap(iris, row_cluster = True, col_cluster = False,
                   method = 'single',metric = 'euclidean',row_colors = row_colors)
g.savefig('Iris-heatmap-Minimum-method.png')

# euclidean distance and maximum cluster distance method for clustering
g = sns.clustermap(iris, row_cluster = True, col_cluster = False,
                   method = 'complete',metric= 'euclidean',  row_colors = row_colors)
g.savefig('Iris-heatmap-maximum-method.png')

