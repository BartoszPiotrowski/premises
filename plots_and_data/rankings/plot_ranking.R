library(ggplot2)
library(reshape)

t <- commandArgs(trailingOnly = TRUE)[1]
r <- read.csv(paste('rankings', t, sep='/'), head=FALSE)
p <- read.csv(paste('proofs', t, sep='/'), head=FALSE)
r$a <- r[,1] %in% p[,1]
colnames(r) <- c("premise", "score", "ATP_useful")
r$premise <- factor(r$premise, levels=r$premise)
#m<-max(as.integer(rownames(r)[r$ATP_useful]))
m<-40
r <- r[1:m,]
ggplot(r, aes(x=premise, y=score)) +
	geom_point(aes(color=ATP_useful), size=6) +
	theme(axis.text.x=element_text(angle=60,hjust=1,vjust=1,size=13)) +
	theme(plot.title = element_text(hjust = 0.5)) +
	ggtitle(paste("Predictions for conjecture", t)) +
	xlab('Premises') +
	ylab('Score')
ggsave(paste('plots/', t, '.pdf', sep=''), device='pdf', width=10)
