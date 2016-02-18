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

## Separate out user data set

users <-  training_data[,grepl("u_|user_id", names(training_data))]
users <-  users[!duplicated(users),]

# Transform user vote types in a single column
userVotes <-  melt(subset(users, select=c(u_cool, u_funny, u_useful)))
names(userVotes) <-  c('vote_type', 'Count')
# removing all null values correspoding to vote type
userVotes <-  userVotes[!is.na(userVotes$Count),]
userVotes$vote_type <-  as.factor(gsub("u_", "", as.character(userVotes$vote_type)))

# EDA1: Boxplots of user votes by vote type/category
usr1 <- ggplot(data=userVotes, aes(x=vote_type, y=Count, fill=vote_type)) + 
  geom_boxplot() +
  ggtitle("Boxplots of Number of User Votes by Vote Type")+ 
  xlab("Vote Type") + 
  ylab("Number of Votes by User Across All Reviews") + coord_flip() +
  guides(fill=FALSE)
usr1 <-  BeautifyPlot(usr1)
usr1

ggsave(usr1, filename="Plots/userVoteTypeBoxplots.png", height=6, width=6, units="in")

# Limit the range of count of votes displayed as distributions are heavily right-skewed
usr1 <-  usr1 + ylim(0,quantile(userVotes$Count, 0.75)) + ggtitle("Boxplots of Number of User Votes by Vote Type
                  (75th Percentile cut off)")
usr1
ggsave(usr1, filename="Plots/userVoteTypeBoxplotsLim.png", height=6, width=6, units="in")


# EDA2: Barchart of number of total votes for each vote category
usr2 <-  ggplot(userVotes, aes(y=Count, x=vote_type, fill=vote_type)) + 
  geom_bar(stat="identity")+ 
  ggtitle("Total User Votes by Vote Type")+ 
  xlab("Vote Type") + 
  ylab("Number of Votes")+
  guides(fill=FALSE)
usr2 <-  BeautifyPlot(usr2)
usr2
ggsave(usr2, filename="Plots/userTotalVoteType.png", height=5, width=5, units="in")


# EDA3: Density Plot of user average ratings
UserAvgRating <-  round(mean(users$u_average_stars, na.rm=TRUE),2)

usr3 <-  ggplot(users, aes(x=u_average_stars)) + 
  geom_density(na.rm=TRUE, colour="blue4", 
                       fill="blue2", alpha=0.2) + 
  ylim(0,1.2)+ 
  geom_vline(xintercept=UserAvgRating)+ 
  geom_text(label=paste("Mean:", signif(UserAvgRating,5)), 
                    x=4.3, y=1.2, size=5)+
  ggtitle("Density Plot of User Average Star Ratings")+ 
  xlab("Average Star Ratings") + 
  ylab('Density') + 
  theme_classic()
usr3 <-  BeautifyPlot(usr3)
usr3
ggsave(usr3, filename="Plots/userAvgRating_Density.png", width=7, height=5, units="in")


# EDA4: Density plot of review count of users for all reviews

RvwCntmedian <-  median(users$u_review_count, na.rm=TRUE) # 7
RvwCntmean <-  mean(users$u_review_count, na.rm=TRUE) # 38.85873
RvwCntmax <-  max(users$u_review_count, na.rm=TRUE) # 5807
                
usr4 <-  ggplot(users, aes(x=u_review_count))+
  geom_density(na.rm=TRUE, colour="BLUE4",fill="blue2", alpha=0.2)+ 
  geom_text(x=2000, y=0.05, size=3, label=paste("Mean Review Count:", signif(RvwCntmean, 5))) +
  ggtitle("Density Plot of User Review Counts")+ 
  xlab("Number of Reviews") + 
  ylab("Density")
usr4
ggsave(usr4, filename="Plots/userReviewCnt_Density.png", width=6, height=4, units="in")


# EDA5: Comparison of review count of users to review count of all known users
rvwTotal <-  data.frame(cbind(c("Reviews by Users in Dataset", "Reviews by Users"),
                             c(sum(!is.na(training_data$u_average_stars)), sum(users$u_review_count, na.rm=TRUE))))
names(rvwTotal) <-  c("rev_source", "num_rvw")
rvwTotals$num <-  as.numeric(as.character(rvwTotal$num_rvw))
usr5 <-  ggplot(rvwTotal, aes(x=rev_source, y=num_rvw)) + 
  geom_bar(stat='identity', fill="blue4") +
  geom_text(aes(label=rvwTotal$num_rvw), vjust=-0.3, size=3.5)+ 
  ggtitle("Reviews by known Users in Reviews Data
          vs. All Reviews by known Users")+ 
  xlab("Review Source") + 
  ylab("Number of Reviews")
usr5 <-  BeautifyPlot(usr5)
usr5
ggsave(usr5, filename="Plots/usrReviewCountComparison.png", width=5, height=5, units="in")


# EDA6: Correltaion between user vote type and average rating of user
userVoteTypeRtng <- users[ !is.na(users$u_average_stars) , c(2,6,7,8)]
names(userVoteTypeRtng) <-  c("Average_Stars", "cool", "funny", "useful")

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

usr6 <- pairs(userVoteTypeRtng, lower.panel = panel.smooth,  diag.panel = panel.hist,
      upper.panel = panel.cor,  main = "Vote Type vs Average Star Rating of User", pch = 21, bg = "blue")
usr6

png("Plots/userVoteTypePairs.png", height=6, width=6, units="in", res=300)


# EDA7: Correlation between user average stars overall and average stars in review dataset
userAvgStarsRvw <-  aggregate(r_stars ~ user_id, data=training_data, mean, na.rm=TRUE)
names(userAvgStarsRvw)[names(userAvgStarsRvw)=="r_stars"] <-  "r_stars_avg"
userAvgStarsRvw <-  merge(userAvgStarsRvw, users[,c('user_id', 'u_average_stars')], all=FALSE)
userAvgStarsRvw <-  subset(userAvgStarsRvw, !is.na(u_average_stars))
corr_uavg_ravg <-  cor(userAvgStarsRvw$u_average_stars, userAvgStarsRvw$r_stars_avg)
usr7 <-  ggplot(userAvgStarsRvw, aes(x=u_average_stars, y=r_stars_avg))+ 
  geom_point() + 
  geom_text(x=0.75, y=4.5, size=3, label=paste("Correlation:", signif(corr_uavg_ravg, 4)))+ 
ggtitle("User Average Stars All Reviews vs. Reviews in Data Set")+ 
  xlab("Average Stars Across All Reviews")+ 
  ylab("Average Stars for Reviews in Data Set")
usr7 <-  BeautifyPlot(usr7)
usr7
ggsave(usr7, filename="Plots/userAvgStarsRvwAllCor.png", width=5, height=5, units="in")



