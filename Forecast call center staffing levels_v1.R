
# install library to read excel files
# install.packages("readxl, dependencies = TRUE")
library(readxl)

data<- read_xlsx(file.choose())

# get the dimension of the data set
dim(data)

# view the top 6 rows of the data set
head(data)

# view the last 6 rows of data
tail(data)

# view the entire data set
view(data)

# examine the structure of the data set
str(data)

# get the statistical summary of each variable
summary(data)

# get "summary tools" library
library("summarytools")

# print the summary table
dfSummary(data)

# check for missing values
colSums(is.na(data))

# Data Cleaning #1: clean the 4,288 missing values in customer satisfaction
# insert "0s" on missing value rows (4,288 rows or 13.6% of total) in customer satisfaction column
# rationale: each of the missing value row had an outcome = "HANG"; customers who hang up 
# receive a "0" customer satisfaction score.
data$customer_satisfaction <- ifelse(data$outcome=="HANG",0,data$customer_satisfaction)

# check customer_satisfaction for missing values
summary(data$customer_satisfaction)

# check for outliers
boxplot(data$customer_satisfaction)

boxplot(data$age)

# Feature Engineering #1: create new feature "handling_time"
# handling_time = how long (in seconds) was the call serviced by the agent
data$handling_time <- data$service_exit - data$service_start

# check the structure of handling_time
dfSummary(data$handling_time)
# handling_time has 1,243 negative values
# handling time has negative values due to the day change which starts at 00:00:01
# adding 86,400 seconds (24*60*60 = 86,400 seconds) will adjust the handling_time to the correct day time value

# Data Cleaning #2: add 86400 seconds to each negative row of handling_time 
data$handling_time <- ifelse(data$handling_time < 0, data$handling_time + 86400, data$handling_time)

summary(data$handling_time)
dfSummary(data$handling_time)
# summary shows handling_time Min value is 0.0; this means all negative values have been transformed.

# Feature Engineering #2: create new variable: years_at_company
data$years_at_company <- 2018 - data$year_of_joining

# check the new variable
summary(data$years_at_company)

# Feature Engineering #3: create new variable: waiting_time
data$waiting_time <- data$service_start - data$server_entry

# Data Cleaning #3: fix discrepancy #1: if outcome = "ANSWERED" when handling_time = "0", then outcome must be changed to outcome = "HANG"
data$outcome <- ifelse(data$handling_time == 0, "HANG", "ANSWERED")

# Data Cleaning #4: fix discrepancy #2: if customer_satisfaction is blank when outcome = "HANG", then customer_satisfaction 
# must be changed to customer_satisfaction = "0"
data$customer_satisfaction <- ifelse(data$outcome == "HANG", 0, data$customer_satisfaction)

# Data Cleaning #5: fix discrepancy #3: if outcome = "HANG", then waiting_time should = server_exit - server_entry
data$waiting_time <- ifelse(data$outcome == "HANG", data$server_exit - data$server_entry, data$waiting_time)

# check to make sure each of the data discrepancies have been corrected:
summary(data$waiting_time)
summary(data$outcome)
dfSummary(data$outcome)
summary(data$customer_satisfaction)
dfSummary(data$customer_satisfaction)

# Data Cleaning #6: waiting_time has negative values due to the day change; fix the problem by adding 86,400 seconds to 
# negative values of waiting_time
data$waiting_time <- ifelse(data$waiting_time < 0, data$waiting_time + 86400, data$waiting_time)

summary(data$waiting_time)
# discrepancy fixed - waiting_time has no more negative values

# Feature Engineering #4: dimension reduction - drop 3 unnecessary variables: "service_exit", "service_start" and "year_of_joining".
data[ ,c('service_exit','service_start','year_of_joining')] <- list(NULL)

dim(data)

view(data)

dfSummary(data$handling_time)
# discrepancy fixed - no negative values

dfSummary(data$years_at_company)
# data checked - no negative values

summary(data$years_at_company)

pie(table(data$shift))
prop.table(table(data$shift))*100
round(prop.table(table(data$shift))*100, digits=1)
# Business Insight #1: Shift 2 is the biggest shift followed by Shift 3

pie(table(data$outcome))
round(prop.table(table(data$outcome))*100, digits=1)
# Business Insight #2: Incoming calls are answered around 86% of the time while customers hang up 14% of the time.

pie(table(data$gender))
round(prop.table(table(data$gender))*100, digits=1)
# Business Insight #3: roughly 77% of sales agents are male, 23% are female.

pie(table(data$priority))
round(prop.table(table(data$priority))*100, digits = 1)
# Business Insights #4: 58% of incoming calls are rated "priority 0" while 28% are assigned a "priority 2" rating.

pie(table(data$ftr))
round(prop.table(table(data$ftr))*100, digits = 1)
# Business Insight #5: around 71% of most issues are resolved in the first call while roughly 29% are not resolved in the first call.

hist(data$age)
# Business Insight #6: most agents are between the age of 25 to 35 years.

hist(data$customer_satisfaction)


hist(data$years_at_company)
# Business Insight #7: most customer satisfaction scores are between 75 to 90

# Count the number of calls per Agent ID
table(data$agent_id)
# Business Insight #8: on average, most agents receive an average of over 2,000 calls per month

# Bivariate Analysis

# generate a list of the average (mean) customer satisfaction by agent id
x<- aggregate(x= data$customer_satisfaction, by= list(data$agent_id), FUN = mean)
x

x<- aggregate(x= data$handling_time, by= list(data$agent_id), FUN = mean)
x

# Use the dplyr library in R

# count the number of incoming calls that went unanswered by each agent
x<- data %>%
  select(agent_id, outcome) %>%
  filter(outcome == "HANG")
x

# print the number of unanswered calls per agent id
table(x$agent_id)

# calculate the average (mean) speed to answer by agent id
dd11<- subset(data,outcome == 'ANSWERED')
x<- aggregate(x=dd11$waiting_time, by= list(dd11$agent_id), FUN=mean)
x
# Business Insights #9: it takes anywhere from 3 to 9 minutes for a call to get answered.

# calculate the average (mean) time the customer waited before hanging up by agent id
dd11<- subset(data,outcome == 'HANG')
x<- aggregate(x=dd11$waiting_time, by= list(dd11$agent_id), FUN=mean)
x
# Business Insights #10: customers wait around 15 to 20 seconds before hanging up.

# count the number of "ftr" events by agent id
table(data$agent_id, data$ftr)
# Agents 1-4 have a higher ratio of negative FTRs
# Agents 10-15 have a higher ratio of positive FTRs

# we now use "age" as the comparative factor against other variables

# generate a list of the average (mean) customer satisfaction by age
x<- aggregate(x= data$customer_satisfaction, by= list(data$age), FUN = mean)
x
# Business Insight #11: experienced agents are better at handling customers. As the agent's age
# increases, customer satisfaction also increases.

# generate a list of the average (mean) handling time by age
x<- aggregate(x= data$handling_time, by= list(data$age), FUN = mean)
x
# Business Insight #12: as the agent's age increases, average handling time also increases.

# Use the dplyr library in R
# count the number of incoming calls that went unanswered by agent's age
x<- data %>%
  select(age, outcome) %>%
  filter(outcome == "HANG")
x

# print the number of unanswered calls by age
table(x$age)
# Business Insight #13: on average, as the agent's age increases, the number of dropped calls decreases.

# calculate the average (mean) time to answer the call by age
dd11<- subset(data,outcome == 'ANSWERED')
x<- aggregate(x=dd11$waiting_time, by= list(dd11$age), FUN=mean)
x
# Business Insight #14: no relation exists between the agent's age and the average time to answer the call

# calculate the average (mean) time the customer waited before hanging up by age
dd11<- subset(data,outcome == 'HANG')
x<- aggregate(x=dd11$waiting_time, by= list(dd11$age), FUN=mean)
x
# Business Insight #15: no relation exists between the agent's age and the average time the customer waited before hanging up

# count the number of "ftr" events by age
table(data$age, data$ftr)

# Business Insight #16: as the agent's age increases, so does the number of positive "FTRs"; newer agents (age: 21-25) could
# benefit from additional training or mentorship programs as they seem to have a higher number of unresolved FTRs.

# Analyze "years at company" or tenure as the comparative factor against other variables

# generate a list of the average (mean) customer satisfaction by tenure
x<- aggregate(x= data$customer_satisfaction, by= list(data$years_at_company), FUN = mean)
x
# Business Insight #17: as agent tenure increases, so does their customer satisfaction scores. 
# Agents who have been with the company longer are better at handling customers.

# generate a list of the average (mean) handling time by years at company
x<- aggregate(x= data$handling_time, by= list(data$years_at_company), FUN = mean)
x
# Business Insight #18: no relation exists between the agent's tenure and average handling time.

# Use the dplyr library in R

# count the number of unanswered calls by tenure
x<- data %>%
  select(years_at_company, outcome) %>%
  filter(outcome == "HANG")
x

# print the number of unanswered calls by agent's tenure
table(x$years_at_company)

# Business Insight #19: as agent tenure increases, the number of dropped calls decreases.
# more experienced agents tend to drop fewer calls than relatively newer agents.

# calculate the average (mean) time to answer the call by tenure.
dd11<- subset(data,outcome == 'ANSWERED')
x<- aggregate(x=dd11$waiting_time, by= list(dd11$years_at_company), FUN=mean)
x
# Business Insight #20: no relation exists between average time to answer the call and agent's tenure.

# calculate the average (mean) time the customer waited before hanging up by tenure
dd11<- subset(data,outcome == 'HANG')
x<- aggregate(x=dd11$waiting_time, by= list(dd11$years_at_company), FUN=mean)
x
# Business Insight #21: Except for agen't with a tenure of 10 years, there doesn't seem to be a relation between 
# the average wait time before hanging up and agent's tenure.

# count the number of "ftr" events by tenure
table(data$years_at_company, data$ftr)
# Business Insight #22: after three years, the number of FTRs substantially improves; the likelihood of customers 
# having their issues resolved in the first call improves greatly with an agent who has 4 years plus tenure.

# Model Deployment: Building the Erlang C Algorithm
# A. Data Preparation

# show column names
names(data)

# Install tidyr package in R for datetime transform
library(tidyr)

# Transform "server_entry" feature: remove date and keep time only
data$server_entry

# create new feature
newDat<- separate(data,server_entry, into = c("date1","server_entry"), sep= " ")
newDat

# Transform "server_exit" feature: remove date and keep time only
newDat<- separate(newDat,server_exit, into = c("date2","server_exit"), sep= " ")
newDat

# drop variables date1 and date2
newDat[ ,c('date1','date2')] <- list(NULL)
newDat

# review the column names in the data set
names(newDat)

# Examine the structure of the date feature
str(data$date)

# Transform "date" feature from a POSIXct to a dateformat data type
newDat$date<- as.Date(newDat$date)
str(newDat$date)

# get a count of all incoming calls by date
table(newDat$date)

# get the mean of daily incoming calls for the time period 10/15/18 to 11/14/18
mean(table(newDat$date))
# On average, the call center gets about 1,088 calls per day

# B. Creating a training data set to train the Erlang C algorithm

# We now build a training data set to train the Erlang C algorithm to the forecast model parameters.
# First, we choose a date from the call center data set, where the average daily call volume closely matches the average monthly call 
# volume of the call center data set. 
# We chose 2018-11-08 as the date because its daily call volume of 1,098 closely matches the average monthly call volume of 1,088.
# Second, We create a training data set "ddd" with 1,098 observations and 16 features from the 11/08/18 data set to train the model's algorithm.

ddd<- newDat[newDat$date=="2018-11-08",]
ddd

# an alternative way to slice the data is to use the subset method in R.
ddf<- subset(newDat,date=="2018-11-08")
ddf

# C. Defining the Model Parameters

# Install "chron" library in R to help objects handle dates and times
library(chron)

# declare a start time variable "z" and an end time variable "z1" in hours, minutes and seconds
z <- times("00:00:00")
z1 <- times("23:59:59")

# add the number of hourly rows by 1 each time you count the number of incoming calls
nrow(ddd[ddd$server_entry >=z & ddd$server_entry <z +(1/24), ])

# there were 9 incoming calls between 12:00 a.m. to 1:00 a.m. on 11/08/18

# === CREATE A LOOP THAT COUNTS ALL HOURLY INCOMING CALLS on 11/08/18 FROM 12:00 a.m. to 11:59 p.m. ====

v11<-vector("numeric")
x=1
while(z<=z1)
{
  v11[x]<-nrow(ddd[ddd$server_entry >=z & ddd$server_entry <z +(1/24), ])
  z=z+(1/24)
  x=x+1
}
v11

# ======= CONSTRUCTING THE ERLANG C ALGORITHM ==========

# SHOW THE ERLANG C FORMULA HERE.


# Goal: Find the number of call center agents that are needed to meet the service level benchmarks.

# DEFINE THE NUMERATOR FOR THE "PW" EQUATION

# Step 1: Declare the Variables for the model

# 1. rate = number of calls coming in
rate<- 200

# 2. duration = average handling time (180 seconds = 3 minutes)
duration<- 180

# 3. target = how many seconds the call needs to be answered by.
target<- 20

#4. gos_target (grade of service target) = the percent of calls that should be answered within the targeted time?
# Note: the industry standard is 80% of calls should be answered within 20 seconds.
gos_target<- 80

#5. interval = the time interval of AHT (average handling time) is in hours; therefore 60 minutes = 1 hour
interval<- 60

#6. declare the formula for intensity "int" (aka erlangs or call hours).
int <- (rate * (duration/interval))/60
# what is the value of "int"?
int
# Intensity is 10 erlangs or 10 call hours.

#7. declare a formula for the number of agents "agents" and round it to a whole number.
agents <- round(int) + 1
agents
# So, with the intensity of 10 erlangs, we would require 11 agents in order to meet the service level target.

#8. Step 2: Declare the numerator for PW as "a":
a<- ((int^agents) * agents)/(factorial(agents) * (agents - int))

# get the value of "a"
a

# DEFINE THE DENOMINATOR FOR THE "PW" EQUATION

#9. Declare the denominator for "PW" as "b":

# Step 3: construct a loop for the summation sign in the denominator

b=1
for(i in 1:agents-1)
{ 
  b<- b+((int^(1))/factorial(1))
}
b

#10. DEFINE THE FORMULA FOR PW
pw<- a/(a+b)

# Solve for PW
pw

# The probability that the caller will wait is 99.6%.


#===== Deploy the PW formula =======

int <- (rate * (duration/interval))/60
agents <- round(int) + 1
a<- ((int^agents) * agents)/(factorial(agents) * (agents - int))
b=1
for(i in 1:agents-1)
{ 
  b<- b+((int^(1))/factorial(1))
}
pw<- a/(a+b)
pw

# the probability of a call wait is 0.995 or 99.5%

#============= Declare the Service Level Formula ==============

# SHOW THE SERVICE LEVEL FORMULA HERE.

# Step 4: Define the formula for service level "SL":

SL<- 1- (pw*exp(-(agents-int)*(target/duration)))
SL
# If the number of agents deployed is 1, then the service level is 10.9%
# This is well below the target service level of 80%
# How do we get 80%? We increment the number of agents by 1

# ==== For a step-wise increase in service level, increment the number of agents by 1 each time ====

int <- (rate * (duration/interval))/60
agents<- round(int)

agents<- agents + 9

a<- ((int^agents) * agents)/(factorial(agents) * (agents - int))
b=1
for(i in 1:agents-1)
{ 
  b<- b+((int^(1))/factorial(1))
}
pw<- a/(a+b)

SL<- 1- (pw*exp(-(agents-int)*(target/duration)))
SL

# Staffing 9 sales agents to the call desk raises the service level to 82.5%

# ==== CREATE A LOOP TO CALCULATE THE NUMBER OF AGENTS (N) TO MEET THE TARGET SL of 80% =====

rate<- 200
duration<- 180
target<- 20
gos_target<- 80
interval<- 60

int<- (rate * (duration/interval))/60
agents<- round(int)+1

while(TRUE)
{
  a<- ((int^agents)*agents)/(factorial(agents)*(agents-int))
  
  b=1
  for(i in 1:agents-1)
  { 
    b<-b+((int^(i))/factorial(i))
  }
  pw<-a/(a+b)
  
  SL<- 1-(pw*exp(-(agents-int)*(target/duration)))
  if(SL>= (gos_target/100))
  {
    break()
  }
  agents<- agents+1
}

agents
SL

# The model shows a staffing level of 14 agents is needed to achieve a SL = 88.8%

#====== Create the RESOURCE FUNCTION to link the Erlang C algorithm to the data table ========

# variables that go into the resource function
rate<- 200
duration<- 180
target<- 20
gos_target<- 80
interval<- 60

# resource tells you how many agents you need for a given target SL.
resource<- function(rate,duration,target,gos_target,interval=60)
{
  int<- (rate * (duration/interval))/60
  agents<- round(int)+1
  
  while(TRUE)
  {
    a<- ((int^agents)*agents)/(factorial(agents)*(agents-int))
    b=1
    for(i in 1:agents-1)
    { 
      b<-b+((int^(i))/factorial(i))
    }
    pw<-a/(a+b)
    
    SL<- 1-(pw*exp(-(agents-int)*(target/duration)))
    if(SL>= (gos_target/100))
    {
      return(c(agents,SL))
    }
    agents<- agents+1
  }
}

resource(200,180,20,80,60)

#======== Link Resource function to data table ========

x=1
y<- vector("numeric")
while(x<=24)
{
  print(resource(v11[x],mean(data$handling_time),20,80,60))
  x=x+1
}

y


