library(ggplot2)
library(reshape)

trees<-read.csv('logs/trees_mptp2078_new.csv')
colnames(trees)<-c("Number of trees", "Percentage of proved theorems")
print(head(trees))
ggplot(trees, aes(x=`Number of trees`, y=`Percentage of proved theorems`)) +
	geom_smooth() + geom_point() +
	theme(axis.text.x=element_text(angle=35,hjust=1,vjust=1)) +
	scale_x_log10(breaks=trees$`Number of trees`)
ggsave('trees.png', device='png', height=3.1, width=3.4)

eta<-read.csv('logs/eta_mptp2078_new.csv')
colnames(eta)<-c("Eta parameter", "Percentage of proved theorems")
print(head(eta))
ggplot(eta, aes(x=`Eta parameter`, y=`Percentage of proved theorems`)) +
	geom_smooth(span = 0.8) + geom_point() +
	#theme(axis.text.x=element_text(angle=40,hjust=1,vjust=1)) +
	scale_x_log10(breaks=eta$`Eta parameter`)
ggsave('eta.png', device='png', height=3.1, width=3)

max_depth<-read.csv('logs/max_depth_mptp2078_new.csv')
colnames(max_depth)<-c("Max depth of trees", "Percentage of proved theorems")
print(head(max_depth))
ggplot(max_depth, aes(x=`Max depth of trees`, y=`Percentage of proved theorems`)) +
	geom_smooth(span = 0.8) + geom_point() +
#	theme(axis.text.x=element_text(angle=40,hjust=1,vjust=1)) +
	scale_x_log10(breaks=max_depth$`Max depth of trees`)
ggsave('max_depth.png', device='png', height=3.1, width=3)
