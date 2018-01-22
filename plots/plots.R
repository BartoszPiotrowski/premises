library(ggplot2)
library(reshape)

trees<-read.csv('plots/trees.csv')
colnames(trees)<-c("Number of trees", "Percentage of proved theorems")
print(head(trees))

ggplot(trees, aes(x=`Number of trees`, y=`Percentage of proved theorems`)) +
	geom_smooth() + geom_point() +
	scale_x_log10(breaks=c(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
			       	2048, 4096, 8192, 16384))
