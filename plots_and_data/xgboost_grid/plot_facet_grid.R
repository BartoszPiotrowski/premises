library(ggplot2)
library(reshape)

trees<-read.csv('logs/trees_mptp2078_new.csv')
eta<-read.csv('logs/eta_mptp2078_new.csv')
max_depth<-read.csv('logs/max_depth_mptp2078_new.csv')

trees$param<-rep('trees', nrow(trees))
eta$param<-rep('eta', nrow(eta))
max_depth$param<-rep('max_depth', nrow(max_depth))

cn<-c('value','thms','param')
colnames(trees)<-cn
colnames(eta)<-cn
colnames(max_depth)<-cn

data<-rbind(trees,eta,max_depth)
print(data)

gg<-ggplot(data, aes(x=value, y=thms)) + geom_point()
#	geom_smooth() +
#	theme(axis.text.x=element_text(angle=40,hjust=1,vjust=1)) +
#	scale_x_log10(breaks=max_depth$`Max depth of tree`)
gg + facet_grid(. ~ param, scales="free_x")
ggsave('grid.png', device='png', height=4, width=9)
