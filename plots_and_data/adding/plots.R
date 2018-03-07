library(ggplot2)
library(reshape)

knn<-read.csv('data/adding_knn_mptp2078_new.csv', header=F)
simple<-read.csv('data/adding_simple_mptp2078_new.csv', header=F)
short<-read.csv('data/adding_short_mptp2078_new.csv', header=F)
negmin_1<-read.csv('data/adding_negmin_1_mptp2078_new.csv', header=F)
negmin_2<-read.csv('data/adding_negmin_2_mptp2078_new.csv', header=F)
negmin_rand<-read.csv('data/adding_negmin_rand_mptp2078_new.csv', header=F)
negmin_all<-read.csv('data/adding_negmin_all_mptp2078_new.csv', header=F)

print(simple)
print(short)
print(simple[simple$V1 < 999,])
data<-data.frame(
	Round=seq(1,30),
	kNN=knn[knn$V1 < 999,][1:30],
	XGB_simple=simple[simple$V1 < 999,][1:30],
	XGB_short=short[short$V1 < 999,][1:30],
	XGB_negmin_1=negmin_1[negmin_1$V1 < 999,][1:30],
      #	XGB_negmin_2=negmin_2[negmin_2$V1 < 999,][1:30],
	XGB_negmin_all=negmin_all[negmin_all$V1 < 999,][1:30],
	XGB_negmin_rand=negmin_rand[negmin_rand$V1 < 999,][1:30]
	)

data_melt<-melt(data, c("Round"))
colnames(data_melt)<-c("Round", "Method", "Number of proved theorems")
print(data_melt)

ggplot(data_melt, aes(x=`Round`, y=`Number of proved theorems`)) +
	geom_point(aes(color=Method, shape=Method), alpha=0.9) +
	geom_line(aes(color=Method), alpha=0.8) +
	scale_y_continuous(breaks=c(230,240,250,260,270,280,290,300,310,320)) +
	scale_x_continuous(breaks=c(3,6,9,12,15,18,21,24,27,30))
#	theme(axis.text.x=element_text(angle=30,hjust=1,vjust=1))
ggsave('addings.png', device='png', height=2.7, width=5.5, dpi=800)
