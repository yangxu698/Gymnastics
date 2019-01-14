##  Name: Yang XU  ##
##  Email: yangxu698@gmail.com  ##
setwd("/home/yang/Lucid")
data.raw = read.csv("data_science_screening_exercise_orders.csv")
data.raw[,1] = as.factor(data.raw[,1])

###  Part A  Data Wrangling ##
library(dplyr)
data_A = data.raw %>% group_by(customer_id)
summary(data_A)

data_A_answer = tbl_df(matrix(0, length(unique(data.raw[,1])), 4))
data_A_answer[,1:2] = unique(data.raw[,1:2])
order_count = count(data_A[,1])

latest_order_date = data_A %>% top_n(1, date)
latest_order_date %>% filter(customer_id == '6750')

data_A_answer = inner_join(latest_order_date, order_count, by = 'customer_id') %>%  select(-value)
colnames(data_A_answer)[3:4] = c("most_recent_order_date", "order_count")
data_A_answer = unique(data_A_answer)
summary(data_A_answer)
head(data_A_answer)


###  Part B  Plot  ###
library(tidyverse)
data.raw = unique(data.raw)
data.raw = data.raw %>% mutate( weeknumber = strftime(data.raw$date, format = "%V")) %>%
                        mutate( weeknumber = ifelse(str_detect(date,'2017-01-01'),'01', weeknumber))
## Given the fact that Jan.01.2017 is Sunday, and the system defaultly assign the weeknumer 52 to it,
## Thus, I manually adjust the weeknumber for this day as 01.
head(data.raw)
data_B = dplyr::count(data.raw,weeknumber)
colnames(data_B)[2] = "order_count"
library(ggplot2)

data_B %>% ggplot(aes(x = weeknumber, y = order_count ))+
           geom_bar(stat = 'identity',fill ='lightblue') +
           theme(axis.text.x = element_text(angle= 90, hjust = 0), plot.title = element_text(hjust=0.5))+
           ggtitle("Order Count on Weeknumber")


###  Part C Significance Test ###
data_C = data.raw %>% group_by(gender) %>% dplyr::summarize(Mean = mean(value))
data_C
var.test(data.raw[,4]~data.raw[,2])

## By variance test, the order value for both genders are statistically different
## thus, t-test is set as var.equal = FALSE
t.test(data.raw[,4]~data.raw[,2], var.equal = F)
## T-test shows the the null hypothesis cannot be rejected given 0.05 cut-off threshold
## Thus the conclusion is: there is no statistical difference between the mean order value
## for both genders
