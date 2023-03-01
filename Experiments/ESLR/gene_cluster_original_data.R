library('factoextra')
library(plotly)
library(dplyr)

df = read.csv('./microarray/xtrain.data', FALSE, sep="")

fviz_nbclust(df, kmeans, method="silhouette")

fviz_nbclust(df, kmeans, method="gap_stat")


pca <- prcomp(df, scale.=TRUE)
cl <- kmeans(df, 4)
cluster = factor(cl$cluster)

pca_df = data.frame(pca$x[,1:3])

p <- plot_ly(pca_df, x=~PC1, y=~PC2, z=~PC3, color=~cluster) %>%
  add_markers(size=1.5)
print(p)
