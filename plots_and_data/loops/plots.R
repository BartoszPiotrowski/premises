library(ggplot2)
library(reshape)

knn<-read.csv('data/loop_knn_mptp2078.csv', header=F)
simple<-read.csv('data/loop_simple_mptp2078_new.csv', header=F)
short<-read.csv('data/loop_short_mptp2078_new.csv', header=F)
negmin_1<-read.csv('data/loop_negmin_1_mptp2078_new.csv', header=F)
negmin_2<-read.csv('data/loop_negmin_2_mptp2078_new.csv', header=F)
negmin_rand<-read.csv('data/loop_negmin_rand_mptp2078_new.csv', header=F)
negmin_all<-read.csv('data/loop_negmin_all_mptp2078_new.csv', header=F)

data<-data.frame(
	Round=seq(0,30),
	kNN=knn[1:31,],
	XGB_simple=simple[1:31,],
	XGB_short=short[1:31,],
	XGB_negmin_1=negmin_1[1:31,],
#	XGB_negmin_2=negmin_2[1:31,],
	XGB_negmin_all=negmin_all[1:31,],
	XGB_negmin_rand=negmin_rand[1:31,]
	)

data_melt<-melt(data, c("Round"))
colnames(data_melt)<-c("Round", "Method", "Number of proved theorems")
print(data_melt)

ggplot(data_melt, aes(x=`Round`, y=`Number of proved theorems`)) +
	geom_point(aes(color=Method, shape=Method), alpha=0.9) +
	geom_line(aes(color=Method), alpha=0.7) +
	scale_y_continuous(breaks=c(300,400,500,600,700,800,900,1000,1100)) +
	scale_x_continuous(breaks=seq(0,30,3))
#	theme(axis.text.x=element_text(angle=31,hjust=1,vjust=1))
ggsave('loops.png', device='png', height=3, width=5, dpi=800)
