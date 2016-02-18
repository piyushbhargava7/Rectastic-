library(ggplot2)
library(reshape2)
library(GGally)
library(scales)

# Code to beautify plots
BeautifyPlot = function(plot){
  # Input: ggplot object
  # Output: same ggplot object with changes in plot theme
  NewPlot = plot + theme(plot.title = element_text(family = "Trebuchet MS", 
                                                   color = "#666666", 
                                                   face = "bold", 
                                                   size = 20,
                                                   vjust = 0.5)) +
    theme(axis.title = element_text(family = "Trebuchet MS", 
                                    color = "#666666", 
                                    face = "bold", 
                                    size = 15,
                                    vjust = 0.5)) +
    theme(panel.border = element_blank(), 
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank(), 
          axis.line = element_line(colour = "black"),
          plot.title = element_text(face="bold")) +
    theme(axis.text.x = element_text(colour = "black",
                                     size = 9)) +
    theme(axis.text.y = element_text(colour = "black",
                                     size = 9))
  
  
  
  return(NewPlot)
}

setwd("/Users/sakshi024/Desktop/University Docs/Advanced Machine Learning/Project/")
#import training data
training_data <- read.csv('yelp_training.csv')

# EDA1: Number of reviews by Star Rating
rvw1 = ggplot(training_data, aes(x=as.factor(r_stars))) + 
  geom_bar(fill='blue4')+ 
  ggtitle("Number of Reviews by Star Rating")+ 
  xlab("Star Rating") + 
  ylab("Number of Reviews")
rvw1 <-  BeautifyPlot(rvw1)
rvw1
ggsave(rvw1, filename="Plots/reviewsDistbyStarRating.png", width=5, height=5, units="in")



# EDA2: Correltaion between review vote type and average rating of user
rvwVoteTypeRtng <- training_data[  , c(4,7,8,9)]
names(rvwVoteTypeRtng) <-  c("Stars", "cool", "funny", "useful")

panel.hist <- function(x, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5) )
  h <- hist(x, plot = FALSE)
  breaks <- h$breaks; nB <- length(breaks)
  y <- h$counts; y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y, col = "orange", ...)
}
panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- abs(cor(x, y))
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
  text(0.5, 0.5, txt, cex = 2.0, col = "dark blue")
}

rvw2 <- pairs(rvwVoteTypeRtng, lower.panel = panel.smooth,  diag.panel = panel.hist,
              upper.panel = panel.cor,  main = "Vote Type vs Star Rating of review", pch = 21, bg = "blue")
rvw2

