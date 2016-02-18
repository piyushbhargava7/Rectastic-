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
#import trainingdata
training_data <- read.csv('yelp_training.csv')

# Get number of checkins per business
checkins <- subset(training_data, select=c(business_id, b_sum_checkins, b_stars))
checkins <-  checkins[!duplicated(checkins),]
checkins <-  checkins[order(checkins$b_sum_checkins, decreasing = TRUE),]


# EDA1: number of checkins per business
chk1 = ggplot(checkins[,], aes(x=b_sum_checkins)) + 
  geom_histogram(fill="blue4")+ 
  ggtitle("Number of Checkins Per Business")+ 
  xlab("Number of Checkins") + 
  ylab("Number of Business")
chk1 <- BeautifyPlot(chk1)
chk1
ggsave(chk1, filename="Plots/checkinsNumBusiness.png", width=5, height=5, units="in")


# calculate median, mean, max and outlier limit
medianChk <- median(checkins$b_sum_checkins, na.rm = TRUE) # 30
meanChk <- mean(checkins$b_sum_checkins, na.rm = TRUE) # 94.29993
MaxChk <-  max(checkins$b_sum_checkins, na.rm = TRUE) # 22977
sdChk <- sd(checkins$b_sum_checkins, na.rm = TRUE) # 314.257
# choosing a lower cut off to remove outliers
cutoff_Chk <- meanChk + 3*sdChk  # 1037.071


# proportion of businesses with less than cutoff checkins
sum(checkins$b_sum_checkins<cutoff_Chk, na.rm=TRUE)/nrow(checkins)

# EDA2: Limit number of checkins by cut off
chk2 = ggplot(checkins[checkins$b_sum_checkins<cutoff_Chk,], aes(x=b_sum_checkins)) + 
  geom_histogram(fill="blue4")+ 
ggtitle("Number of Checkins Per Business
                  (Less than 1037 checkins)") + 
  xlab("Number of Checkins") +
  ylab("Frequency")
chk2 <- BeautifyPlot(chk2)
chk2
ggsave(chk2, filename="Plots/checkinsNumBusinessLim.png", width=5, height=5, units="in")


# EDA3: Businesses With and Without Checkins
checkins$has_checkins <-  !is.na(checkins$b_sum_checkins)
chk3 <-  ggplot(checkins, aes(x=has_checkins)) + 
  geom_bar(stat="bin", fill="blue4") +
  ggtitle("Businesses With and Without Checkins")+ 
  xlab("No Checkin = False, Checkin = True") + 
  ylab("Number of Businesses")
chk3 <-  BeautifyPlot(chk3)
chk3
ggsave(chk3, filename="Plots/checkinsBusYesNo.png", width=5, height=5, units="in")


# EDA4: Number of reviews against number of checkins (in data set)
checkins$b_sum_checkins[is.na(checkins$b_sum_checkins)] <-  0
NumReviewsPerBus <- aggregate(. ~ business_id, training_data[,1:2], FUN=length)

names(NumReviewsPerBus)[2] <- 'num_of_reviews'

checkins <-  merge(checkins, NumReviewsPerBus)
corr_chkrvw <-  cor(checkins$b_sum_checkins, checkins$num_of_reviews)


chk4 <- ggplot(checkins[checkins$b_sum_checkins<MaxChk,], 
            aes(x=b_sum_checkins, y=num_of_reviews)) + 
  geom_point() + 
  ggtitle("Business Checkins vs. Reviews in Dataset")+ 
  xlab("Number of Checkins") + 
  ylab("Number of Reviews") + 
  geom_text(x=3500, y=600, size=4, label=paste("Correlation:", signif(corr_chkrvw, 4)))

chk4 <- BeautifyPlot(chk4)
chk4

ggsave(chk4, filename="Plots/checkinsVsReviewsCorr.png", width=5, height=5, units="in")


