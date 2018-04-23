###   R software tutorial for beginners   ###
###   https://goo.gl/ADKSbt   ###
#               7.14.2017                  #

# load the data 
# make sure to set the working directory
# setwd("/Users/wyang/Documents/R/marin")


LungCapData <- read.csv("LungCap.CSV")
View(LungCapData)

ls(LungCapData)
summary(LungCapData)


# to check the Age variable
LungCapAge <- LungCapData$Age
View(LungCapAge)


###     1.9 Logic Statements (TRUE/FALSE) and cbind and rbind Command in R         ###

# To compare whether the age > 15, 
# and return the result into a new variable Age_15
LungCapData$Age_15 <- LungCapData$Age>15
View(LungCapData)

# to save the data as numerical
LungCapData$Age_15_1 <- as.numeric(LungCapData$Age_15)


# to take a look at the first 3 observations
LungCapData[1:3,]



# multiple filter to answeer multiple questions
FemSmoke <- LungCapData$Smoke == "yes" & 
                          LungCapData$Gender == "female"
FemSmoke

# cbind & rbind
MoreData <- cbind(LungCapData,FemSmoke)
MoreData

# To remove all contents in the working space
#rm(list = ls())



###   1.10 How to Set Up a Working Directory in R    ###
# To check the working directory
getwd()

# setwd("/Users/wyang/Documents/R/marin")

# to save the progress
save.image("LungCap_1.R")
rm(list = ls())

# to load the save project R file
load("LungCap_1.R")

# to load the save project R file manually
load(file.choose())



###   1.11 Scripts for Reproducible Research in R    ###
# The shortcut to comment : ctrl + shift + c, or
# Code --> Comment/Uncomment Lines 

# Tab key after the first several letters when you type
class(LungCapData$Gender)



###   1.12 How to Install Packages in R    ###
help(install.packages)

install.packages("epiR")

# Or Install package manually
# Tools -->Install packages

library(epiR)

help(package = epiR)



###   1.13 Customizing The Look of R Studio    ###
# Tools -->Globle options



###========== 2 Statistics =======###

### 2.1 How to Make Bar Charts and Pie Charts in R ###

# Barcharts and Piecharts are appropriate for summarizing 
# the distribution of a categorical variable

LungCapData <- read.csv("LungCap.CSV")

dim(LungCapData)
names(LungCapData)
class(LungCapData$Gender)

# A Barchart is a visual display of the frequency for each 
# caterory of a categorical variable or the relative frequency(%)
# for each category

help("barplot")

count <- table(LungCapData$Gender)
count

percent <- table(LungCapData$Gender)/725
percent

barplot(count)

barplot(percent, main="Title", xlab="Gender", ylab="%", las=1, names.arg=c("Female","Male"))


# piechart
help("piechart")

pie(count, main="Title")

box()



### 2.2a How to Make Boxplots and Boxplots With Groups in R  ###

# A boxplot is appropriate for summarizing the distribution 
# of a numeric variable

help("boxplot")

boxplot(LungCapData$LungCap, main="Title", 
        xlab = "LungCap", ylab= "Lung Capacity", ylim=c(0,16))

quantile(LungCapData$LungCap, probs=c(0,0.25,0.5,0.75,1))

# slice LungCap by gender
boxplot(LungCapData$LungCap ~ LungCapData$Gender, main="Lung Capacity vs Gender", 
        xlab = "Gender", ylab= "Lung Capacity", ylim=c(0,16))

# subset the data
boxplot(LungCapData$LungCap[LungCapData$Gender == "female"],
        LungCapData$LungCap[LungCapData$Gender == "male"])

########
# =(equal) sign is used in R to assign values to objects
# ==(double equal) sign is used in R to represent the meaning of equality in a mathematical sense!
########



### 2.2b How to Produce Stratified Boxplots in R  ###

# Stratified boxplots are useful for examing the relationship between a cateforical variable and 
# a numeric variable, within stata or groups defined by a third categorical variable

# Exam the relationship between Smoking and Lung Capaciry within Age groups or Age strata
# Create an "AgeGroups" variable (Age strata)
AgeGroups <- cut(LungCapData$Age, breaks=c(0,12,15,20), labels=c("<12","12/15","16+"))
AgeGroups[1:5]

levels(AgeGroups)
summary(AgeGroups)

# to compare the lung capacity between smokers and non-smokers
boxplot(LungCapData$LungCap ~ LungCapData$Smoke, main="Lung Capacity vs Smoke", 
        ylab="Lung Capacity", las=1 )

# to compare the lung capacity between smokers and non-smokers who are older than 14 years 
boxplot(LungCapData$LungCap[LungCapData$Age >= 14] ~ LungCapData$Smoke[LungCapData$Age >= 14], 
        main="Lung Capacity vs Smoke older than 14", ylab="Lung Capacity", las=1 )

# Visualize the relationship between Lung Capacity and smoking within each of the age strata
# create boxplots of Lung capacity for smokers and non-smokers for: 12 or younger; 13-15, 16 or older
boxplot(LungCapData$LungCap ~ LungCapData$Smoke*AgeGroups, main="Lung Capacity vs Smoke, by Age Group", 
        ylab="Lung Capacity", las=2 )

# to color different groups, color #4=blue, #2=red
boxplot(LungCapData$LungCap ~ LungCapData$Smoke*AgeGroups, main="Lung Capacity vs Smoke, by Age Group", 
        ylab="Lung Capacity", las=2, col=c(2:7) )

#add box around it
box()

#relable y-axis, x-axis and legend
axis(2, at=seq(0,20,4), seq(0,20,4),las=2)
axis(1,at=c(1.5,3.5,5.5), labels=c("<12","12-15","16+"))
 legend(x=4.5, y=4.5, legend=c("Non-Smoke","Smoke", col=c(4,2)))



###       2.3 How to Make Histograms in R      ###
# A histogram is appropriate for summarizing the distribution of a numeric variable
help(hist)

head(LungCapData)

hist(LungCapData$LungCap)

# to change the y-axis "frequency" to the "density" 
hist(LungCapData$LungCap,freq=FALSE)

# to set the bin width
hist(LungCapData$LungCap,prob=T, ylim=c(0,0.16), breaks = 10)

hist(LungCapData$LungCap,prob=T, ylim=c(0,0.16), breaks = seq(from=0, to=20,2))

# to fit the curve,line color(col=24), line width(lwd=3)
lines(density(LungCapData$LungCap),col=2,lwd=3)



###       2.4 How to Make Stem and Leaf Plots in R      ###
# Stem and leaf plots are appropriate for summarizing the distribution of a 
# numeric variable and are most appropriate for smaller datasets

# extract the lung capacity for only females ans save in femLungCap
femLungCap <- LungCapData$LungCap[LungCapData$Gender=="female"]
stem(femLungCap)



###       2.5 How to Make Stacked Bar Charts, Grouped Bar Charts and Mosaic Plots in R      ###
# These plots are appropriate for examing the relationship between 2 categorical variables

names(LungCapData)
class(LungCapData$Gender)
class(LungCapData$Smoke)
levels(LungCapData$Smoke)

# to examine the relationship between Gender and Smoking
# need to produce contingency tables
table1 <- table(LungCapData$Smoke, LungCapData$Gender)
table1

barplot(table1)
barplot(table1,beside=T, legend.text = T, col=c(4, 2))

# to generate the mosaic plot
mosaicplot(table1, col=c(2,3))


###       2.6 How to Make Scatterplots in R      ###
# scatterplots are appropriate for examing the relationship between 2 numerical variables
## Pearson's correlation is used to examine the strenghth of the Linear relationship between 2 numeric variables

# Exploring the relationship between Height and Age

# pearson's correlation
cor(LungCapData$Age, LungCapData$Height)
# xlim, cex=0.5, 50% the size of original scattering
plot(LungCapData$Age, LungCapData$Height, main="Scatterplot", xlab="Age",ylab="Height", xlim=c(0,25),cex=0.5, col=4)

plot(LungCapData$Age, LungCapData$Height, main="Scatterplot", xlab="Age",ylab="Height", xlim=c(0,25),pch=8)

# to add the linear regression
abline(lm(LungCapData$Height ~ LungCapData$Age), col=2)

# line smoother, ty(line type), lwd(line width)
lines(smooth.spline(LungCapData$Age, LungCapData$Height), lty=2, lwd=4)



###  2.7  How to Calculate Mean, Standard Deviation, Frequencies in R  ###
# Categorical Variable(Smoke ) and Numeric Variable(Lung Capacity)
# Caterorical Variable are summarized using a Frequency or a Proportion

table(LungCapData$Smoke)

length(LungCapData$Smoke)

table(LungCapData$Smoke)/length(LungCapData$Smoke)

table(LungCapData$Smoke, LungCapData$Gender)

# Numeric variable
mean(LungCapData$LungCap)
#trim=0.10, remove the top 10% and the bottom 10% of the observation
mean(LungCapData$LungCap,trim=0.10)

median(LungCapData$LungCap); var(LungCapData$LungCap); sd(LungCapData$LungCap)
sqrt(sd(LungCapData$LungCap)); sd(LungCapData$LungCap)^2; min(LungCapData$LungCap)
max(LungCapData$LungCap); range(LungCapData$LungCap); quantile(LungCapData$LungCap)
quantile(LungCapData$LungCap, probs=c(0,0.1,0.5,0.9,1)); sum(LungCapData$LungCap)

cor(LungCapData$LungCap, LungCapData$Age)
cor(LungCapData$LungCap, LungCapData$Age, method="spearman")
cov(LungCapData$LungCap, LungCapData$Age) # covariance
var(LungCapData$LungCap, LungCapData$Age)

summary(LungCapData); summary(LungCapData$LungCap); summary(LungCapData$Age)



###  3.1  Binomial Distribution in R  ###
# Calculating probabilities for binomial random variables using R

# x is binomial distribution with n=20 trials and p=1/6 probability of success 
#    x ~ BIN(n=20, p=1/6)
# dbinom command is used to find values for the probability density function of X, f(X)
# p(x=3), 3 success in 20 trials
dbinom(x=3,size=20, prob=1/6)  # 0.2378866

# p(x=0) & P(x=1)&...& p(x=5)
dbinom(x=0:5,size=20, prob=1/6) # 0.02608405 0.10433621 0.19823881 0.23788657 0.20220358 0.12941029

# p(x<=5)
sum(dbinom(x=0:5,size=20, prob=1/6))  # 0.8981595

# pbinom command gives us values for the probability distribution function of X,f(x)
pbinom(q=5,size=20,prob=1/6, lower.tail=T)  # 0.8981595 
                                            
# rbinom command: to take a random sample from  a binomial distribution 
# qbinom command: to find quantiles for a binomial distribution



###   3.2  Poisson Distribution in R  ###
# Calculating probabilities for poisson random variables using R

# x follows a poisson distribution with a known rate of λ=7 
#    x ~ POISSON(λ=7)
# dpois command: to find values for the probability density function of X, f(X)

# p(x=4), 
dpois(x=4, lambda=7)  # 0.09122619

# p(x=0) & P(x=1)&...& p(x=4)
dpois(x=0:4,lambda=7) #  0.000911882 0.006383174 0.022341108 0.052129252 0.091226192

# p(x<=4)
sum(dpois(x=0:4,lambda=7))  # 0.1729916

# ppois command:returns probability associated with the distribution function of F(x)
ppois(q=4,lambda=7, lower.tail=T)  #0.1729916

# p(x>=12)
ppois(q=12, lambda=7,lower.tail=F) # 0.02699977

# rpois command: to take a random sample from  a poisson distribution 
# qpois command: to find quantiles for a poisson distribution



###  3.3  Normal Distribution, Z Scores, and Normal Probabilities in R  ###
# Calculating probabilities, percentiles and taking random samples from 
# a normally distributed variables using R

# x is Normally distributed with a known mean of 75 and standard deviation of 5
# x ~ N(μ=75, σ^2 =5^2)

# p(x<=70)
pnorm(q=70,mean=75,sd=5,lower.tail = T) # 0.1586553

# p(x>=85)
pnorm(q=85,mean=75,sd=5,lower.tail = F) # 0.02275013

# p(Z>=1)
pnorm(q=1,mean=0,sd=1,lower.tail = F) # 0.1586553

#qnorm can be used to calculate the quantile of normal distribution
qnorm(p=0.25, mean=75,sd=5,lower.tail = T) # Find Q1: 71.62755

# dnorm: find and plot the probability density function

x <- seq(from=55, to=95, by=0.25)
dens <- dnorm(x, mean=75,sd=5)
plot(x,dens)
plot(x,dens,type="l")
plot(x,dens,type="l", main="x-norm:mean=75,sd=5", xlab="x", ylab="probability density", las=1)

abline(v=75)

# rnorm: to draw samples from a known normally distributed function
rand <- rnorm(n=40,mean=75,sd=5)

hist(rand)



###       3.4   t Distribution and t Scores in R       ###
# Finding probabilities and percentilees for t-distribution using R
# These can be used to find p-values or critical values for constructing confidence intervals
# for statistics that follow a t-distribution...
# t follows a t-distribution, with mean=0, standard deviation=1,and 25 degrees of freedom
# t ~ tdf=25, μ=ο, σ=1

# t-stat=2.3, df=25 # one-sided p-value   #p(t>2.3)
pt(q=2.3,df=25,lower.tail = F)   # 0.01503675
# two-sided p-value
pt(q=2.3,df=25,lower.tail = F) + pt(q = -2.3,df=25,lower.tail = T)  # 0.03007351

# find t for 95% confidence == # value of t with 2.5% in each tail: -2.059539
qt(p=0.025,df=25, lower.tail = T)  #  -2.059539

# pf :f-distribution
# pexp: exponential distribution



###     4.1 One-Sample t Test in R    ###
# conducting one-sample t-test and constructing one-sample confidence interval for the mean
# one-sample t-test and confidence interval are parametric methods appropriate for examing a single numeric variable...
LungCapData <- read.csv("LungCap.CSV")
names(LungCapData)
class(LungCapData$LungCap)

# it's useful to plot the data before t-test, boxplot or histogram
boxplot(LungCapData$LungCap)
# Ho: mu>=8, Ha: mu<8
# one-sided 95% CI for mu
t.test(LungCapData$LungCap, mu=8,alternative = "less", conf.level = 0.95)

# two-sided
t.test(LungCapData$LungCap, mu=8,alt = "two.sided", conf = 0.95)

Test<-t.test(LungCapData$LungCap, mu=8,alt = "two.sided", conf = 0.99)
attributes(Test)
Test$conf.int
Test$p.value


###     4.2   Two-Sample t Test in R: Independent Groups  ####
# These are parametric methods appropriate for examing the difference in means for 2 populations
# There are ways of examing the relationship between a numeric outcome variable(Y) and a categorical explanatory variable(X,with 2 levels)

names(LungCapData)
class(LungCapData$LungCap)
class(LungCapData$Smoke)
levels(LungCapData$Smoke)
boxplot(LungCapData$LungCap ~ LungCapData$Smoke)

# Ho:mean lung cap of smokers = of non smokers
# two-sided t test
#Assume non-equal variances

t.test(LungCapData$LungCap ~ LungCapData$Smoke, mu=0,alt="two.sided",conf=0.95,var.eq=F,paired = F)

t.test(LungCapData$LungCap ~ LungCapData$Smoke)

var(LungCapData$LungCap[LungCapData$Smoke == "yes"])
  
var(LungCapData$LungCap[LungCapData$Smoke == "no"])

# Levene's Test
# Ho: population variances are equal

library(car) # the package car(Companion to Applied Regression) needs to be installed
levene.test(LungCapData$LungCap ~ LungCapData$Smoke)


###   4.3  Mann Whitney U aka Wilcoxon Rank-Sum Test in R    ###
# Conducting the Mann-Whitney U Test A.K.A Wilcoxon Rank-Sum Test
# This is a non-parametric method appropriate for examing the difference in Medians for 2 independent populations...
# It's a way of examing the relationship between a numeric outcome variable(Y) and a categorical explanatory variable(X, with 2 levels) when groups are independent!

class(LungCapData$LungCap);    levels(LungCapData$Smoke)
boxplot(LungCapData$LungCap ~ LungCapData$Smoke)
# Ho:Median Lung Capacity of smokers =  that of non smokers
# two-sided test
wilcox.test(LungCapData$LungCap ~ LungCapData$Smoke, mu=0, alt="two.sided", conf.int=T,conf.level=0.95, paired=F,exact=F, correct=T)



###    4.4   Paired t Test in R    ###
# Conducting the paired t-test and Confidence Interval Using R
# These are parametric methods appropriate for examining the difference in Means for 2 populations that are paired or dependent on one another...
BP<-read.csv("BloodPressure.csv")
boxplot(BP$Before,BP$After)
# To check the distribution of the data
plot(BP$Before,BP$After)
abline(a=0,b=1)  # 45' line
#Ho:Mean difference in SBP=0
# two-sided
t.test(BP$Before,BP$After, mu=0,alt="two.sided",paired=T,conf.level = 0.99)



###     4.5  Wilcoxon Signed Rank Test in R  ###
# This is a non-parametric method appropriate for examining the Median difference in observations for 2 populations that are paired or dependent on one another...
BP<-read.csv("BloodPressure.csv")
boxplot(BP$Before,BP$After)

#Ho:Median change in SBP=0
#two-sided
wilcox.test(BP$Before,BP$After, mu=0,alt="two.sided",paired=T, conf.int=T, conf.level=0.99,exact=F, correct=F)



###   4.6   Analysis of Variance (ANOVA) and Multiple Comparisons in R   ###
# ANOVA is a parametric method appropriate for comparing the Means for 2 or more independent populations
DietData <- read.csv("DietWeightLoss.CSV")
class(DietData)
names(DietData)
boxplot(DietData$WeightLoss ~ DietData$Diet)
# Ho: Mean weight loss is the same for all diets
aov(DietData$WeightLoss ~ DietData$Diet)
ANOVA1 <- aov(DietData$WeightLoss ~ DietData$Diet)

summary(ANOVA1)
attributes(ANOVA1)
ANOVA1$coefficients

TukeyHSD(ANOVA1)

plot(TukeyHSD(ANOVA1),las=1)


## Kruskal Wallis One-way Analysis of Variance is a non-parametric equivalent ot the one-way Analysis of Variance
kruskal.test(DietData$WeightLoss ~ DietData$Diet)



###   4.7  Chi-Square Test, Fishers Exact Test, and Cross Tabulations in R  ###
# The chi-square test of independence is a parametric method appropriate for testing independence between two cateforical variables
LungCapData <- read.csv("LungCap.CSV")
table(LungCapData$Gender, LungCapData$Smoke)
TAB <-table(LungCapData$Gender, LungCapData$Smoke)
barplot(TAB,beside=T, legend=T)
chisq.test(TAB,correct=T)
CHI=chisq.test(TAB,correct=T)
attributes(CHI)
CHI$expected

# Fisher's Exact test is a non-parametric alternative to the Chi-Square test
fisher.test(TAB, conf.int=T, conf.level = 0.99)



###  4.8  Relative Risk, Odds Ratio and Risk Difference (aka Attributable Risk) in R  ###
# These are all measures of the direction and the strength of the association between two categorical variables
# Explore the relationship between Gender and Smoking
TAB<-table(LungCapData$Gender, LungCapData$Smoke)
barplot(TAB, beside=T,legend=T)
#install.packages(epiR)
install.packages("survival")
library(epiR)
epi.2by2(TAB, method="cohort.count", conf.level=0.95) # method: cohort or case study

# to generate the 2x2 table using the matrix function
TAB2 <- matrix(c(44, 314, 33, 334), nrow=2, byrow=T)
TAB2

#table combine
TAB3<-cbind(TAB[,2],TAB[,1])
TAB3

#To give the column names
colnames(TAB3)<-c("yes","no")
TAB3
# to calculate the epi.2by2 again
epi.2by2(TAB3, method="cohort.count")



###   4.9 How to Calculate Correlation and Covariance in R  ###
# Pearson's correlation is parametric measure of the linear association between 2 numeric variables
# Spearman's rank correlation is a non-parametric measure of the monotonic association between 2 numeric variables
# Kendall's rank correlation is another non-parametric measure of the association, based on concordance or discordance of x-y pairs

# Explore the relationship between Age and Lung Capacity
# Set the working directory and Inport the data
LungCapData <- read.csv("LungCap.CSV")
plot(LungCapData$Age, LungCapData$LungCap, main="scatter plot",las=1)
cor(LungCapData$Age, LungCapData$LungCap, method="pearson");cor(LungCapData$Age, LungCapData$LungCap);cor(LungCapData$LungCap,LungCapData$Age)
cor(LungCapData$Age, LungCapData$LungCap, method="spearman")
cor(LungCapData$Age, LungCapData$LungCap, method="kendall")
cor.test(LungCapData$Age, LungCapData$LungCap, method="pearson")
cor.test(LungCapData$Age, LungCapData$LungCap, method="spearman")
cor.test(LungCapData$Age, LungCapData$LungCap, method="spearman", exact=F)
cor.test(LungCapData$Age, LungCapData$LungCap, method="pearson",alt="greater",conf.level=0.99)
cov(LungCapData$Age, LungCapData$LungCap)
pairs(LungCapData)
pairs(LungCapData[,c(1:3)])
cor(LungCapData[,c(1:3)])
cor(LungCapData[,c(1:3)], method="spearman")
cov(LungCapData[,c(1:3)])



###   5.1  Linear Regression in R   ###
# Simple linear regression is useful for examining or modelling the relationship between 2 numeric variables
# we can also fit a simple linera regression using a categorical explanatory (X) variable...
# Model the relationship between Age and Lung Capacity
LungCapData <- read.csv("LungCap.CSV")
plot(LungCapData$Age, LungCapData$LungCap, main="scatter plot",las=1)
cor(LungCapData$Age, LungCapData$LungCap)
MOD<-lm(LungCapData$LungCap ~ LungCapData$Age) # Y ~ X
summary(MOD)
attributes(MOD)
MOD$coefficients
MOD$coef
coef(MOD)
plot(LungCapData$Age, LungCapData$LungCap, main="scatter plot",las=1)
abline(MOD, col=2,lwd=4)
coef(MOD,level=0.99)
anova(MOD)
sqrt(2.3)



###   5.2  Checking Linear Regression Assumptions in R   ###
# While the assumptions of a linear model are never perfectly met, we must still check if they are reasonable assumptions to work with!
# Examining the assumptions for a model of the relationship between Age and Lung Capacity!  Lung Capacity = Y, outcome, dependent variable
plot(LungCapData$Age, LungCapData$LungCap)
MOD <- lm(LungCapData$LungCap ~ LungCapData$Age)
summary(MOD)
abline(MOD)
plot(MOD)

par(mfrow=c(2,2))
plot(MOD)

par(mfrow=c(1,1))
# How non-constant variance will show up in a residual plot
XY <- read.csv("xy.CSV")
plot(XY$x,XY$y)
MOD2 <- lm(XY$y ~ XY$x)
abline(MOD2)

par(mfrow=c(2,2))
plot(MOD2)



###  5.3  Multiple Linear Regression in R  ###
# Multiple linear regression is useful for modeling the relationship between a numeric outcome or dependent variable (Y) and multiple explanatory or independent variables (X)
# Fit a model using Age and Height as X-variables
model1 <- lm(LungCapData$LungCap ~ LungCapData$Age + LungCapData$Height)
# get a summary of the model
summary(model1)

# calculate pearson's correlation between Age and Height
cor(LungCapData$Age, LungCapData$Height, method="pearson")

# ask for confidence intervals for the model coeeficients
confint(model1, conf.level=0.95)

#fit a model using all X variables
model2 <- lm(LungCapData$LungCap ~ LungCapData$Age + LungCapData$Height + LungCapData$Smoke + LungCapData$Gender + LungCapData$Caesarean)
summary(model2)
# check the regression diagnostic plots for this model
plot(model2)



###   5.4  Changing a Numeric Variable to Categorical Variable in R  ###
# Useful for making some cross-tabulations for a variable, or fitting a regression model when the linearity assumption is not valid for the variable, or...
?cut

# we will creat height catefories of A=<50, B=50-55, C=55-60, D=60-65, E=65-70, F=70+
CatHeight <- cut(LungCapData$Height, breaks=c(0,50,55,60,65,70,100), labels=c("A","B", "C","D","E","F"), right=T)
# by default, the intervals are left-open and right-closed,eg. (a,b], meaning that "border" observations go into the left interval...
# it can be change by "right=FALSE" 
# It's also important to specific the labels
LungCapData$Height[1:10]
CatHeight[1:10]








###  5.7  Categorical Variables or Factors in Linear Regression in R  ###
# When building a regression model, one may include numeric variables, categorical variables, or a combination of both as the explanatory or independent (X) variables...
# Explanatory (Independent) variables of Age and Smoke
class(LungCapData$Smoke)
levels(LungCapData$Smoke)

# fit a reg model, using Age+Smoke
model1 <- lm(LungCapData$LungCap ~ LungCapData$Age + LungCapData$Smoke)
summary(model1)

# plot the data ,using different color for smoke(red)/non-smoke(blue)
# first, plot the data for the non-smokers, in blue
plot(LungCapData$Age[LungCapData$Smoke=="no"], LungCapData$LungCap[LungCapData$Smoke=="no"], col="blue",ylim=c(0,15), xlab="Age", ylab="LungCap", main="LungCap vs. Age, Smoke")
# now add the points for the smokers, in solid red circles
points(LungCapData$Age[LungCapData$Smoke=="yes"], LungCapData$LungCap[LungCapData$Smoke=="yes"], col="red", pch=16)
#add in a legend
legend(3,15,legend=c("NonSmoker","Smoker"),col=c("blue","red"), pch=c(1,16),bty="n")
# let's add in the regression lines using the Abline command, blue for non-smoker, red for smoker
abline(a=1.08, b=0.555, col="blue", lwd=3)
abline(a=0.431, b=0.555, col="red",lwd=3)



### 5.8 Categorical Variables in Linear Regression in R, Example #2  ###
# including a categorical variable into a regression model in R, Example #2
# fit a reg model, using Age + CatHeight
model2 <- lm(LungCapData$LungCap ~ LungCapData$Age + CatHeight)
summary(model2)
# plot the data using different color for different Height Categories
plot(LungCapData$Age[CatHeight=="A"],LungCapData$LungCap[CatHeight=="A"], col=2, xlim=c(0,20),ylim=c(0,15),xlab="Age", ylab="LungCap", main="LungCap vs. Age, CatHeight")
# add points of other Height categories
points(LungCapData$Age[CatHeight=="B"],LungCapData$LungCap[CatHeight=="B"], col=3)
points(LungCapData$Age[CatHeight=="C"],LungCapData$LungCap[CatHeight=="C"], col=4)
points(LungCapData$Age[CatHeight=="D"],LungCapData$LungCap[CatHeight=="D"], col=5)
points(LungCapData$Age[CatHeight=="E"],LungCapData$LungCap[CatHeight=="E"], col=6)
points(LungCapData$Age[CatHeight=="F"],LungCapData$LungCap[CatHeight=="F"], col=7)
#add a legend
legend(0,15.5,legend=c("A","B","C","D","E","F"), col=2:7,pch=1, cex=0.8)
# add regression line using abline
abline(a=0.98, b=0.2,col=2, lwd=3)
abline(a=2.46, b=0.2,col=3, lwd=3)
abline(a=3.67, b=0.2,col=4, lwd=3)
abline(a=4.92, b=0.2,col=5, lwd=3)
abline(a=5.99, b=0.2,col=6, lwd=3)
abline(a=7.52, b=0.2,col=7, lwd=3)



### 5.9 Multiple Linear Regression with Interaction in R  ###
# if X1 and X2 interact, this means that the effect of X1 on Y dependents on the value of X2, and vice versa
# examining the interaction between Age and Smoke
# plot the data ,using different color for smoke(red)/non-smoke(blue)
# first, plot the data for the non-smokers, in blue
plot(LungCapData$Age[LungCapData$Smoke=="no"], LungCapData$LungCap[LungCapData$Smoke=="no"], col="blue",ylim=c(0,15), xlab="Age", ylab="LungCap", main="LungCap vs. Age, Smoke")
# now add the points for the smokers, in solid red circles
points(LungCapData$Age[LungCapData$Smoke=="yes"], LungCapData$LungCap[LungCapData$Smoke=="yes"], col="red", pch=16)
#add in a legend
legend(3,15,legend=c("NonSmoker","Smoker"),col=c("blue","red"), pch=c(1,16),bty="n")
# let's add in the regression lines using the Abline command, blue for non-smoker, red for smoker

# fit a reg model, using Age, Smoke, and their INTERACTION
model1 <- lm(LungCapData$LungCap ~ LungCapData$Age * LungCapData$Smoke) # interaction using "*" or ":"
coef(model1)
# add the reg line using abline
abline(a=1.052, b=0.498, col="blue", lwd=3)
abline(a=1.278, b=0.558, col="red",lwd=3)

# ask for that model summary again
summary(model1)

# fit the model that does NOT include INTERACTION
model2 <- lm(LungCapData$LungCap ~ LungCapData$Age + LungCapData$Smoke)
summary(model2)



###   5.10  Interpreting Interaction in Linear Regression   ###
# we will examine the concept of interaction and how this is interpreted for 2 categorical "X" variables with 2 levels each...

#### 5.11 Variable Selection in Linear Regression Using Partial F-Test in R ###
# The partial F-test is used in model building and variable selection to help decide if a variable or term can be removed from a model without making the model significantly worse...
# we can also think of the test as helping us decide if adding a variable or term to the model makes it significantly better...
# the lager model = Full Model
# the model with one or more variable/terms removed = Reduced Model
# Partial F-test is used to compare Nested models
LungCapData <- read.csv("LungCap.CSV")

#  Example #1 
# suppose we have a model to estimate the mean Lung Capacity using Age, Gender, Smoke, and Height
# Full Model: Y= b0 + b1Age + b2Gender + b3Smoke + b4Height
# Reduced Model: Y= b0 + b1Age + b2Gender + b3Smoke

# Example #2
FullModel <- lm(LungCapData$LungCap ~ LungCapData$Age + I((LungCapData$Age)^2))
ReducedModel <-lm(LungCapData$LungCap ~ LungCapData$Age)

summary(FullModel)
summary(ReducedModel)

# Carry out the partial F-test
anova(ReducedModel, FullModel)

# Example #1
model1 <- lm(LungCapData$LungCap ~ LungCapData$Age + LungCapData$Smoke + LungCapData$Gender + LungCapData$Height)
summary(model1)

model2 <- lm(LungCapData$LungCap ~ LungCapData$Age + LungCapData$Smoke + LungCapData$Gender)
summary(model2)

anova(model1, model2)



###  5.12 Polynomial Regression in R  ###
# Polynomial Regression is a special case of linear regression where the relationship between X and Y is modeled using a polynomial, rather than a line
# It can be used when the relationship between X and Y is nonlinear, although this is still considered to be a special case of Multiple Linear Regression.

# read the data
LungCapData <- read.csv("LungCapData2.CSV")
# make a plot of LungCap vs. Height
plot(LungCapData$Height, LungCapData$LungCap, main="Polynomial Regression", las=1)

# now, let's fit a linear regression
model1 <- lm(LungCapData$LungCap ~ LungCapData$Height)
summary(model1)
# and add the line to the plot...make it thick and red...
abline(model1, lwd=3, col="red")

# first, the WRONG WAY...
model2 <- lm(LungCapData$LungCap ~ LungCapData$Height + (LungCapData$Height^2))
summary(model2)

# now, the RIGHT WAY...
model2 <- lm(LungCapData$LungCap ~ LungCapData$Height + I(LungCapData$Height^2))
summary(model2)

# or, create Height^2, and then include this in model...it's the same!
HeightSquare <- (LungCapData$Height)^2
model2again <- lm(LungCapData$LungCap ~ LungCapData$Height + HeightSquare)
summary(model2again)

# or, use the "poly" command...it's the same!
model2againagain <- lm(LungCapData$LungCap ~ poly(LungCapData$Height, degree=2, raw=T))
summary(model2againagain)

# let's remind ourseles of the polynomial model we fit
summary(model2)

# now, let's add this model to the plot, using a thick blue line
lines(smooth.spline(LungCapData$Height, predict(model2)), col="blue", lwd=3)

# test if the model including Height^2 i signif. better than one without
# using the partial F-test
anova(model1, model2)

# try fitting a model that includes Height^3 as well
model3 <- lm(LungCapData$LungCap ~ LungCapData$Height + I((LungCapData$Height^2)) + I((LungCapData$Height^3)))
summary(model3)

# now, let's add this model to the plot, using a thick dashed green line
lines(smooth.spline(LungCapData$Height, predict(model3)), col="green", lwd=3, lty=3)

# and, let's add a legend to clarify the lines
legend(46, 15, legend = c("model1: linear", "model2: poly x^2", "model3: poly x^2 + x^3"), 
       col=c("red", "blue", "green"), lty=c(1,1,3), lwd=3, bty="n", cex=0.9)

# let's test if the model with Height^3 is signif better than one without
anova(model2, model3)

















