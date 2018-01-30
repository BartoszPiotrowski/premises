library(ggplot2)
library(reshape)

knn<-read.csv('data/loop_knn_mptp2078.csv', head=F)
simple<-read.csv('data/loop_simple_mptp2078_new.csv', head=F)
short<-read.csv('data/loop_short_mptp2078_new.csv', head=F)
negmin_1<-read.csv('data/loop_negmin_1_mptp2078_new.csv', head=F)
negmin_2<-read.csv('data/loop_negmin_2_mptp2078_new.csv', head=F)
negmin_rand<-read.csv('data/loop_negmin_rand_mptp2078_new.csv', head=F)
negmin_all<-read.csv('data/loop_negmin_all_mptp2078_new.csv', head=F)

print(knn)
print(simple)
print(short)

data<-data.frame(
	Round=seq(0,14),
	kNN=knn[1:15,],
	simple=simple[1:15,],
	short=short[1:15,],
	negmin_2=negmin_2[1:15,],
	negmin_all=negmin_all[1:15,]
	)

data_melt<-melt(data, c("Round"))
colnames(data_melt)<-c("Round", "Method", "Number of proved theorems")
print(data_melt)

ggplot(data_melt, aes(x=`Round`, y=`Number of proved theorems`)) +
	geom_point(aes(color=Method), alpha=0.9) +
	geom_line(aes(color=Method), alpha=0.6) +
	scale_x_continuous(breaks=data_melt$Round) +
#	theme(axis.text.x=element_text(angle=40,hjust=1,vjust=1))
ggsave('loops.pdf', device='pdf', height=6, width=7, dpi=500)
