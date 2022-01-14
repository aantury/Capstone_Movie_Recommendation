if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
#-------------------------------------------------------------------------------

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

#-------------------------------------------------------------------------------

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]      
temp <- movielens[test_index,]      

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>%              
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

#-------------------------------------------------------------------------------

# Train and Test set for models
#As Validation set is 10% of movielens dataset, train and test dataset can be created
#in a similar way but this time test dataset will be 10% of edx data 

set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_data <- edx[-test_index,]      
temp <- edx[test_index,]      

# Make sure userId and movieId in the test set are also in the train set
test_data <- temp %>% 
  semi_join(train_data, by = "movieId") %>%
  semi_join(train_data, by = "userId")
# Add rows removed from the test set back into the train set
removed <- anti_join(temp, test_data)
train_data <- rbind(train_data, removed)

#-------------------------------------------------------------------------------

#Function to compute RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#-------------------------------------------------------------------------------

#1. Basic Model (Average calculation model)
mu_hat <- mean(train_data$rating)
mu_hat

model_1_rmse <- RMSE(test_data$rating, mu_hat)
model_1_rmse

# Results table
rmse_results <- tibble(Method = "1. Average Calculation Model", RMSE = model_1_rmse)

rmse_results %>% knitr::kable()
#-------------------------------------------------------------------------------

#2. Adding movie effect
mu_hat <- mean(train_data$rating)
movie_averages <- train_data %>%
  group_by(movieId) %>%
  summarise(b_hat_i = mean(rating - mu_hat))

movie_averages

predicted_ratings <- mu_hat + test_data %>%
  left_join(movie_averages, by="movieId") %>%
  pull(b_hat_i)

model_2_rmse <- RMSE(test_data$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="2. Movie effect Model",
                                     RMSE = model_2_rmse ))
rmse_results %>% knitr::kable()

#-------------------------------------------------------------------------------

#3. Adding user effect 
mu_hat <- mean(train_data$rating)
user_averages <- train_data %>%
  left_join(movie_averages , by="movieId") %>%
  group_by(userId) %>%
  summarise(b_hat_u = mean(rating - mu_hat- b_hat_i))

predicted_ratings <- test_data %>%
  left_join(movie_averages, by="movieId") %>%
  left_join(user_averages, by="userId") %>%
  mutate(pred = mu_hat + b_hat_i + b_hat_u) %>%
  pull(pred)

model_3_rmse <- RMSE(test_data$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="3. Movie + User effect Model",
                                     RMSE = model_3_rmse ))
rmse_results %>% knitr::kable()

#-------------------------------------------------------------------------------

#5 Regularised Model with movie effect

#Cross validation to select λ for regularised movie effect model

lambdas <- seq(0, 10, 0.25)

mu_hat <- mean(train_data$rating)
just_the_sum <- train_data %>% 
  group_by(movieId) %>% 
  summarise(s = sum(rating - mu_hat), n_i = n())

rmses <- sapply(lambdas, function(l){
  predicted_ratings <- test_data %>% 
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_hat_i = s/(n_i+l)) %>%
    mutate(pred = mu_hat + b_hat_i) %>%
    pull(pred)
  return(RMSE(test_data$rating, predicted_ratings))
})

lambdas[which.min(rmses)]
qplot(lambdas, rmses)

#-------------------------------------------------------------------------------

lambda <- 1.5 #This value is derived from the cross validation.

mu_hat <- mean(train_data$rating)
movie_regularised_averages <- train_data %>% 
  group_by(movieId) %>% 
  summarise(b_hat_i = sum(rating - mu_hat)/(n()+lambda), n_i = n())

predicted_ratings <- test_data %>% 
  left_join(movie_regularised_averages, by = "movieId") %>%
  mutate(pred = mu_hat + b_hat_i) %>%
  pull(pred)
RMSE(predicted_ratings, test_data$rating)

model_4_rmse <- RMSE(test_data$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="4. Regularised Movie effect Model",
                                     RMSE = model_4_rmse ))

rmse_results %>% knitr::kable()

#-------------------------------------------------------------------------------

#6 Regularised Model with movie and user effect

#Cross validation to select λ for regularised movie + user effect model

lambdas <- seq(0, 10, 0.25)  

rmses <- sapply(lambdas, function(l){
    mu_hat <- mean(train_data$rating)
     b_hat_i <- train_data %>% 
     group_by(movieId) %>%
     summarise(b_hat_i = sum(rating - mu_hat)/(n()+l))
  
     b_hat_u <- train_data %>% 
     left_join(b_hat_i, by="movieId") %>%
     group_by(userId) %>%
     summarise(b_hat_u = sum(rating - b_hat_i - mu_hat)/(n()+l))
  
    predicted_ratings <- test_data %>% 
     left_join(b_hat_i, by = "movieId") %>%
     left_join(b_hat_u, by = "userId") %>%
     mutate(pred = mu_hat + b_hat_i + b_hat_u) %>%
     pull(pred)
  
  return(RMSE(test_data$rating, predicted_ratings))
})

lambdas[which.min(rmses)]
qplot(lambdas, rmses)  

#-------------------------------------------------------------------------------



lambda <- 5  #This value is derived from the cross validation.

mu_hat <- mean(train_data$rating)
regularised_user_averages <- train_data %>%
  left_join(movie_averages , by="movieId") %>%
  group_by(userId) %>%
  summarise(b_hat_u = sum(rating - mu_hat - b_hat_i)/(n()+lambda), n_i = n())

predicted_ratings <- test_data %>%
  left_join(movie_regularised_averages, by = "movieId") %>%
  left_join(regularised_user_averages, by ="userId") %>%
  mutate(pred = mu_hat + b_hat_i + b_hat_u) %>%
  pull(pred)
RMSE(predicted_ratings, test_data$rating)

model_5_rmse <- RMSE(test_data$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="5. Regularised Movie + User effect Model",
                                     RMSE = model_5_rmse ))

rmse_results %>% knitr::kable()

#-------------------------------------------------------------------------------

#7. Recommendation model with Matrix Factorisation using the Recosystem package. 
#Note: Methodology used is based on Recosystem documentation 
#page https://rdocumentation.org/packages/recosystem/versions/0.5

library(recosystem)
memory.limit(size=30000) #This is to ensure that the memory size is adequate

#-------------------------------------------------------------------------------
#a. Create a model object using train and test datasets. Data sets to be converted 
#to Recosystem format

#Train set: This is an object of class DataSource originaing from train_data dataset 
#and the Recosystem function used is data_memory() (source https://www.rdocumentation.org/packages/recosystem/versions/0.5)
train_dataset <- with(train_data,data_memory(user_index = userId, 
                                             item_index = movieId, rating = rating, index1 = TRUE))

#Test set: This is an object of class DataSource originaing from train_data dataset 
#and the Recosystem function used is data_memory() (source https://www.rdocumentation.org/packages/recosystem/versions/0.5)
test_dataset <- with(test_data, data_memory(user_index = userId, 
                                            item_index = movieId, rating = rating, index1 = TRUE))
#A model object with a Reference Class object in R
r = Reco()

#-------------------------------------------------------------------------------

#b. Using default tuning parameters. The default values have been selected as per 
#Rdocumentation webpage https://www.rdocumentation.org/packages/recosystem/versions/0.5/topics/tune
#therefore, the $tune function is not necessary and will not be used  

#-------------------------------------------------------------------------------

#c. Train the model

r$train(train_dataset)

#-------------------------------------------------------------------------------

#d. Compute predicted values

predicted_ratings = r$predict(test_dataset, out_memory())
model_6_rmse <- RMSE(test_data$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="6. Matrix Factorisation (Recosystem) model",
                                     RMSE = model_6_rmse ))

rmse_results %>% knitr::kable()

#-------------------------------------------------------------------------------

#By looking at the different models used so far, it can be concluded that the best 
#performing model is the one using the Recosystem package. This model will be used 
#to test the result using the Validation set (final hold-out test set)

#a. Create a model object using edx and validation sets. Data sets to be converted 
#to Recosystem format

#Train set
edx_dataset <- with(edx, data_memory(user_index = userId, item_index = movieId, rating = rating, index1 = TRUE))

#Test set
validation_dataset <- with(validation, data_memory(user_index = userId, item_index = movieId, rating = rating, index1 = TRUE))

r = Reco()

#-------------------------------------------------------------------------------

#b. Using default tuning parameters. The default values have been selected as per 
#Rdocumentation webpage https://www.rdocumentation.org/packages/recosystem/versions/0.5/topics/tune
#therefore, the $tune function is not necessary and will not be used 

#-------------------------------------------------------------------------------

#c. Train the model

r$train(edx_dataset)

#-------------------------------------------------------------------------------

#d. Compute predicted values

predicted_ratings = r$predict(validation_dataset, out_memory())
model_7_rmse <- RMSE(validation$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          data_frame(Method="7. Final Validation Matrix Factorisation (Recosystem) model",
                                     RMSE = model_7_rmse ))

rmse_results %>% knitr::kable()

#End of script  
#--------------------------------------------------------------------------------

  
    
