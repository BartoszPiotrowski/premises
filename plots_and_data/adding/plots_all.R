library(ggplot2)
library(reshape)

#knn<-read.csv('data/adding_knn_mptp2078.csv', header=F)
simple<-read.csv('data/adding_simple_mptp2078_new.all.csv', header=F)
short<-read.csv('data/adding_short_mptp2078_new.all.csv', header=F)
#negmin_1<-read.csv('data/adding_negmin_1_mptp2078_new.all.csv', header=F)
negmin_2<-read.csv('data/adding_negmin_2_mptp2078_new.all.csv', header=F)
#negmin_rand<-read.csv('data/adding_negmin_rand_mptp2078_new.all.csv', header=F)
negmin_all<-read.csv('data/adding_negmin_all_mptp2078_new.all.csv', header=F)

print(simple)
print(short)
data<-data.frame(
	Round=seq(0,29),
#	kNN=knn[1:30,],
	XGB_simple=simple[simple$V1 > 2300,][1:30],
	XGB_short=short[short$V1 > 2300,][1:30],
#	XGB_negmin_1=negmin_1[1:30,],
	XGB_negmin_2=negmin_2[negmin_2$V1 > 2300,][1:30],
	XGB_negmin_all=negmin_all[negmin_all$V1 > 2300,][1:30]
#	XGB_negmin_rand=negmin_rand[1:30,]
	)

data_melt<-melt(data, c("Round"))
colnames(data_melt)<-c("Round", "Method", "Number of proved theorems")
print(data_melt)

ggplot(data_melt, aes(x=`Round`, y=`Number of proved theorems`)) +
	geom_point(aes(color=Method), alpha=0.9) +
	geom_line(aes(color=Method), alpha=0.8) +
	#scale_y_continuous(breaks=c(300,300,500,600,700,800,900,1000,1100)) +
	scale_x_continuous(breaks=data_melt$Round)
#	theme(axis.text.x=element_text(angle=30,hjust=1,vjust=1))
ggsave('addings_all.png', device='png', height=6, width=9, dpi=800)
