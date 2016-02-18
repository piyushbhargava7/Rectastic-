library(ggplot2)
library(reshape2)
library(scales)


# Code to beautify plots
BeautifyPlot <-  function(plot){
  # Input: ggplot object
  # Output: same ggplot object with changes in plot theme
  NewPlot <-  plot + theme(plot.title = element_text(family = "Trebuchet MS", 
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
                                     size = 6))
 
  
  
  return(NewPlot)
}

setwd("/Users/sakshi024/Desktop/University Docs/Advanced Machine Learning/Project/")
#import training data
training_data <- read.csv('yelp_training.csv')

#subset for only business columns
business <- training_data[,grepl("b_|business_id", names(training_data))]

# EDA1: number of reviews per business 
NumReviewsPerBus <- aggregate(. ~ business_id, business[,1:2], FUN=length)

names(NumReviewsPerBus)[2] <- 'num_of_reviews'

bus1 <- ggplot(NumReviewsPerBus, aes(x=num_of_reviews)) + 
  geom_density(colour="blue4", fill="blue2", alpha=0.2) +
  ggtitle("Number of Reviews Per Business")+ 
  xlab("Number of Reviews") +
  ylab("Density") +
  theme_classic()
bus1 <- BeautifyPlot(bus1) 
bus1

ggsave(bus1, filename="Plots/businessReviewCount_density.png", width=5, height=5, units="in")

# calculate median, mean, max and outlier limit
medianRvw <- median(NumReviewsPerBus$num_of_reviews) # 6
meanRvw <- mean(NumReviewsPerBus$num_of_reviews) # 19.9278
MaxRvw <-  max(NumReviewsPerBus$num_of_reviews) # 844
sdRvw <- sd(NumReviewsPerBus$num_of_reviews) # 42.8571
# Since the standard deviation is so high choosing a lower cut off to identify outliers
cutoff_Rvw <- meanRvw + sdRvw  # 62.78489

# EDA2: Subsetting Reviews for high end outliers in number of reviews per business
NumReviewsPerBusSub <- NumReviewsPerBus[NumReviewsPerBus$num_of_reviews< cutoff_Rvw, ]
bus2 <- ggplot(NumReviewsPerBusSub, aes(x=num_of_reviews)) + 
  geom_density(colour="blue4", fill="blue2", alpha=0.2) + 
  ggtitle("Number of Reviews Per Business with Less Than 63 Reviews")+ 
  xlab("Number of Reviews") + 
  ylab("Density") +
  theme_classic()
bus2 <- BeautifyPlot(bus2)
bus2
ggsave(bus2, filename="Plots/businessReviewCountLess63.png", width=7, height=5, units="in")


# EDA3: number of reviews per business (count of reviews)
bus3 <- ggplot(NumReviewsPerBusSub, aes(x=num_of_reviews)) + 
  geom_histogram(colour = "white", fill='blue4', binwidth=50)+ 
  xlim(0, 800) + 
  ggtitle("Number of Reviews per Business") + 
  xlab("Number of Reviews") + 
  ylab("Number of Businesses")
bus3 <- BeautifyPlot(bus3)
bus3
ggsave(bus3, filename="Plots/businessReviewCount_Number.png", width=5, height=5, units="in")

# Get distinct business IDs
distinctBus <- business[ ! duplicated( business[ 'business_id' ] ) , ]

# Calculating min and max stars for any business
min_star <- min(distinctBus$b_stars)
max_star <- max(distinctBus$b_stars)

# EDA4: Number of business per star rating
bus4 <- ggplot(distinctBus, aes(x=b_stars)) + 
  geom_histogram(colour = "white", fill='blue4', binwidth = 0.5) +
  xlim(min_star, max_star) + 
  ggtitle("Number Businesses per Business Star Rating") + 
  xlab("Business Star Ratings") + 
  ylab("Number of Businesses") 
bus4 <- BeautifyPlot(bus4)
bus4
ggsave(bus4, filename="Plots/businessStarRatings.png", width=5, height=5, units="in")

# EDA4: Number of Businesses per city
busPerCity <- aggregate(. ~ b_city, distinctBus[,c(1,3)], FUN=length)
names(busPerCity)[2] <- "num_of_bus"
busPerCity <- busPerCity[order(-busPerCity$num_of_bus),]
bus6 <- ggplot(busPerCity, aes(y=num_of_bus, x=b_city)) + 
  geom_bar(stat="identity", fill='blue4') + 
  ggtitle("Number of Businesses per City") + 
  xlab("Cities") + 
  ylab("Number of Businesses") +
  theme(axis.text.x=element_text(angle = 45, hjust=1)) 
bus6 <- BeautifyPlot(bus6)
bus6
ggsave(bus6, filename="Plots/NumBusinessCities.png", width=7, height=5, units="in")

# EDA4: Average Star Rating for cities
city_avg_StarRating <- aggregate(. ~ b_city, distinctBus[, c(3,10)], mean)
names(city_avg_StarRating)[2] <- "avg_star_rating"
meanAvgRating <- round(mean(city_avg_StarRating$avg_star_rating) , 2)
bus7 <- ggplot(city_avg_StarRating, aes(y=avg_star_rating, x=b_city)) +
  geom_bar(stat="identity", fill='blue4')+ 
  ggtitle("Average Star Rating Per city")+ 
  xlab("Cities") + 
  ylab("Average Star Rating")+ 
  geom_hline(aes(yintercept = meanAvgRating, 
                 colour = "darkred")) +
  coord_flip() +
  guides(colour=FALSE)
bus7 <- BeautifyPlot(bus7)
bus7
ggsave(bus7, filename="Plots/AverageRatingPerCity.png", width=7, height=5, units="in")

# Get businessid, star rating and category columns
buscategories <- distinctBus[,-c(2,3,4,5,6,7,8,519)]

# Get count of Businesses per Category in a Dataframe
categories <- buscategories[,-c(1,2,3)]
NumBusinessPerCategory <- colSums(categories)
NumBusPerCat <- melt(NumBusinessPerCategory)
NumBusPerCat$category <-  row.names(NumBusPerCat)
row.names(NumBusPerCat) <-  NULL
names(NumBusPerCat) <- c('num_business', 'category')
NumBusPerCat <- NumBusPerCat[order(-NumBusPerCat$num_business),]
NumBusPerCat$category <- gsub('b_categories_', "", NumBusPerCat$category)

# EDA5: plot the number of businesses per category
bus8 <- ggplot(NumBusPerCat, aes(y=num_business, x=category)) + 
  geom_bar(stat="identity", fill='blue4') + 
  ggtitle("Number of Businesses Per Category")+ 
  xlab("Categories") + 
  ylab("Number of Businesses") +
  theme(axis.text.x=element_text(angle = 45, hjust=1)) 
bus8 <- BeautifyPlot(bus8)
bus8


# We want to know the top categories by number of business
# mean and median on businesses per category
# calculate median, mean, max and outlier limit
medianBusCat <- median(NumBusPerCat$num_business) # 12
meanBusCat <- mean(NumBusPerCat$num_business) # 60.80906
MaxBusCat <-  max(NumBusPerCat$num_business) # 4503
sdBusCat<- sd(NumBusPerCat$num_business) # 242.1728

# EDA6: Top 50 category by number of business
Cat50 <- NumBusPerCat[1:50,]
bus10 <- ggplot(Cat50, aes(y=num_business, x=category)) + 
  geom_bar(stat="identity", fill='blue4') + 
  ggtitle("Top 50 Categories by Number of Businesses")+ 
  xlab("Categories") + 
  ylab("Number of Businesses")+ 
  coord_flip() +
  guides(colour=FALSE)
bus10 <- BeautifyPlot(bus10)
bus10
ggsave(bus10, filename="Plots/businessCatTop50.png", width=7, height=5, units="in")


# EDA7: Total Number of businesses open and closed
busOpen <- distinctBus[,c('business_id', 'b_open')]
NumOpenCLoseBus <- aggregate(.~b_open, busOpen, FUN=length)
names(NumOpenCLoseBus)[2] <- "num_business"
bus14 <- ggplot(NumOpenCLoseBus, aes(y=num_business, x=b_open)) +
  geom_bar(stat="identity", fill='blue4') +
  ggtitle("Number of Businesses Open or Closed") + 
  xlab("Closed = False, Open = True") + 
  ylab("Number of Businesses") +
  theme_classic()
bus14 <- BeautifyPlot(bus14)
bus14
ggsave(bus14, filename="Plots/businessOpenClose.png", width=5, height=5, units="in")

# subset data for open or closed businesses 
openBusiness <- business[business$b_open=="True",]

closeBusiness <- business[business$b_open=="False",]

# EDA8: open and close business star ratings
# Plotting histgram side by side
levels(business$b_open)[levels(business$b_open)=="True"] <- "Open"
levels(business$b_open)[levels(business$b_open)=="False"] <- "Close"

bus15 <- ggplot(business, aes(x = b_stars, fill = b_open))+ 
  geom_histogram(aes(y = ..count..), binwidth = 0.5) +
  ggtitle("Number of businessess by open status") + 
  xlab("Star Ratings") + 
  ylab("Number of Businesses")+
  facet_grid(.~b_open ) +
  theme(strip.text.x = element_text(size = 12, face ="bold"),
        strip.background = element_rect(colour ="red", 
                                        fill = "#CCCCFF")) +
  guides(fill = guide_legend(title = NULL)) 
bus15 <- BeautifyPlot(bus15)
bus15
ggsave(bus15, filename="Plots/OpenCloseBusinessStarRatings.png", width=7, height=5, units="in")

# Business star rating (b_stars) across all reviews (in business dataset)
# vs. avg star rating for the business in review dataset
busAvgStars <-  aggregate(r_stars ~ business_id, data=training_data, mean, na.rm=TRUE)
names(busAvgStars)[names(busAvgStars)=="r_stars"] <-  "r_stars_avg"
busAvgStars <-  merge(busAvgStars, training_data[,c('business_id', 'b_stars')], all=FALSE)

corr_rating <-  cor(busAvgStars$b_stars, busAvgStars$r_stars_avg)

bus16 <- ggplot(busAvgStars, aes(x=b_stars, y=r_stars_avg)) + 
  geom_point() +
  geom_text(x=1.5, y=4.5, size=3, label=paste("Correlation:", signif(corr_rating, 4))) + 
ggtitle("Business Average Star Rating All Reviews vs. Reviews in Data Set") + 
  xlab("Average Star Rating Across All Reviews") + 
  ylab("Average Star Rating Across Reviews in Data Set")
bus16 <-  BeautifyPlot(bus17)
bus16
ggsave(bus16, filename="Plots/BusinessStarRatingCorr.png", width=5, height=5, units="in")

