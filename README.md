<p><strong> Scalable Recommendation System for Yelp </p>

<p>Piyush Bhargava, Sakshi Bhargava, Chhavi Choudhury</p>

[Data Source - RecSys Challenge 2013: Yelp business rating prediction](https://www.kaggle.com/c/yelp-recsys-2013)

<p>Accurately predicting user preference is a difficult
task, however getting reasonable predictions is necessary for
myriad of industries to corroborate personalization, which
is highly useful both economically and socially. Inspired by
this, we chose to build our own recommendation system at
scale, leveraging data from RecSys2013: Yelp Business Rating
Prediction contest. Given a location and a business category,
we aimed to recommend a business to a user that they have not
reviewed before. The recommendations are based on predictions
made using Yelp review ratings, business and user features. This
is essentially a cold start problem as most of the users and
businesses in the test data are absent from the training data.
After creating our baseline model using average ratings, we
applied various modeling techniques such as Random Forest,
Logistic Regression, Linear Regression and Gradient Boosting
on features extracted from metadata to predict business ratings.
We evaluated our models and tested for combination of features
across models using 3 fold cross validation by minimizing
Mean Square Error (MSE). The best RMSE seen on test
data was 1.267 with Random Forest model which is almost
6% improvement over mean baseline of 1.39. We also created
an elementary recommendation tool for users. Building our
recommendation system in Spark ensures scalability for future
applications. In future, we would like to augment our data
(using Yelp API) to decrease sparsity and improve prediction
performance.</p>
