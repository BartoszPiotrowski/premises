library(ggplot2)
library(reshape)

ratios<-read.csv('ratios_mptp2078.csv')
colnames(ratios)<-c('Ratio negatives/positives', 'Percentage of proved theorems')
print(ratios)
ggplot(ratios, aes(x=`Ratio negatives/positives`, y=`Percentage of proved theorems`)) +
geom_smooth(span = 9) + geom_point() +
         #theme(axis.text.x=element_text(angle=40,hjust=1,vjust=1)) +
         scale_x_log10(breaks=ratios$`Ratio negatives/positives`)
ggsave('ratios.png', device='png', height=3.1, width=3, dpi=800)
warnings()
