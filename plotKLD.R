library(ggplot2)
library(reshape2)
kld <- melt(read.csv('KLD.csv'), value.name='KLD', id.vars='Iteration', variable.name='Direction')

ggplot(data=kld, aes(x=Iteration,y=KLD)) + 
  geom_line(aes(colour = Direction))