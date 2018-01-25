library(ggplot2)
library(reshape)

trees<-read.csv('logs/trees_mptp2078_new.csv')
colnames(trees)<-c("Number of trees", "Percentage of proved theorems")
print(head(trees))
ggplot(trees, aes(x=`Number of trees`, y=`Percentage of proved theorems`)) +
	geom_smooth() + geom_point() +
	theme(axis.text.x=element_text(angle=40,hjust=1,vjust=1)) +
	scale_x_log10(breaks=trees$`Number of trees`)
ggsave('trees.pdf', device='pdf', height=4, width=4)

eta<-read.csv('logs/eta_mptp2078_new.csv')
colnames(eta)<-c("Eta parameter", "Percentage of proved theorems")
print(head(eta))
ggplot(eta, aes(x=`Eta parameter`, y=`Percentage of proved theorems`)) +
	geom_smooth() + geom_point() +
	theme(axis.text.x=element_text(angle=40,hjust=1,vjust=1)) +
	scale_x_log10(breaks=eta$`Eta parameter`)
ggsave('eta.pdf', device='pdf', height=4, width=4)

max_depth<-read.csv('logs/max_depth_mptp2078_new.csv')
colnames(max_depth)<-c("Max depth of tree", "Percentage of proved theorems")
print(head(max_depth))
ggplot(max_depth, aes(x=`Max depth of tree`, y=`Percentage of proved theorems`)) +
	geom_smooth() + geom_point() +
	theme(axis.text.x=element_text(angle=40,hjust=1,vjust=1)) +
	scale_x_log10(breaks=max_depth$`Max depth of tree`)
ggsave('max_depth.pdf', device='pdf', height=4, width=4)
