df = read.csv('./gene_embedding_data.csv')

df = t(df)
install.packages('factoextra')
library('factoextra')


fitted.x = fitted(cl)
fviz_nbclust(df, kmeans, method="silhouette")
#fviz_nbclust(df, kmeans, method="gap_stat")

cl <- kmeans(df, 4)
fviz_cluster(cl, data=df)

pca <- prcomp(df, scale.=TRUE)

cluster = factor(cl$cluster)
library(plotly)
library(dplyr)

pca_df = data.frame(pca$x[,1:3])

p <- plot_ly(pca_df, x=~PC1, y=~PC2, z=~PC3, color=~cluster) %>%
  add_markers(size=1.5)
print(p)
