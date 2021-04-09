Running Python Models in R
================

``` r
library(reticulate)
```

## Prerequisites

For these methods to work, you will need to point to a Python executable
in a Conda environment or Virtualenv that contains all the Python
packages you need. You can do this by using a `.Rprofile` file in your
project directory. See the contents of the `.Rprofile` file in this
project to see how I have done this.

## Write Python functions to run on a data set in R

In the file `python_functions.py` I have written the required functions
in Python to perform an XGBoost model on an arbitrary data set. We
expect all the parameters for these functions to to be in a single dict
called `parameters`. I am now going to source these functions into R so
they become R functions that expect equivalent data structures.

``` r
source_python("python_functions.py")
```

## Example: Using XGBoost in R

We now use these Python function on a prepared wine dataset in R to try
to learn to predict a high quality wine.

First we download data sets for white wines and red wines.

``` r
white_wines <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
                        sep = ";")
red_wines <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", 
                      sep = ";")
```

We will create ‘white versus red’ as a new feature, and we will define
‘High Quality’ to be a quality score of seven or more.

``` r
library(dplyr)

white_wines$red <- 0
red_wines$red <- 1

wine_data <- white_wines %>% 
  bind_rows(red_wines) %>% 
  mutate(high_quality = ifelse(quality >= 7, 1, 0)) %>% 
  select(-quality)

knitr::kable(head(wine_data))
```

| fixed.acidity | volatile.acidity | citric.acid | residual.sugar | chlorides | free.sulfur.dioxide | total.sulfur.dioxide | density |   pH | sulphates | alcohol | red | high\_quality |
|--------------:|-----------------:|------------:|---------------:|----------:|--------------------:|---------------------:|--------:|-----:|----------:|--------:|----:|--------------:|
|           7.0 |             0.27 |        0.36 |           20.7 |     0.045 |                  45 |                  170 |  1.0010 | 3.00 |      0.45 |     8.8 |   0 |             0 |
|           6.3 |             0.30 |        0.34 |            1.6 |     0.049 |                  14 |                  132 |  0.9940 | 3.30 |      0.49 |     9.5 |   0 |             0 |
|           8.1 |             0.28 |        0.40 |            6.9 |     0.050 |                  30 |                   97 |  0.9951 | 3.26 |      0.44 |    10.1 |   0 |             0 |
|           7.2 |             0.23 |        0.32 |            8.5 |     0.058 |                  47 |                  186 |  0.9956 | 3.19 |      0.40 |     9.9 |   0 |             0 |
|           7.2 |             0.23 |        0.32 |            8.5 |     0.058 |                  47 |                  186 |  0.9956 | 3.19 |      0.40 |     9.9 |   0 |             0 |
|           8.1 |             0.28 |        0.40 |            6.9 |     0.050 |                  30 |                   97 |  0.9951 | 3.26 |      0.44 |    10.1 |   0 |             0 |

Now we set our list of parameters (a list in R is equivalent to a dict
in Python):

``` r
params <- list(
  input_cols = colnames(wine_data)[colnames(wine_data) != 'high_quality'],
  target_col = 'high_quality',
  test_size = 0.3,
  random_state = 123,
  subsample = (3:9)/10, 
  xgb_max_depth = 3:9,
  colsample_bytree = (3:9)/10,
  xgb_min_child_weight = 1:4,
  k = 3,
  k_shuffle = TRUE,
  n_iter = 10,
  scoring = 'f1',
  error_score = 0,
  verbose = 1,
  n_jobs = -1
)
```

Now we are ready to run our XGBoost model with 3-fold cross validation.
First we split the data:

``` r
split <- split_data(df = wine_data,  parameters = params)
```

This produces a list, which we can feed into our scaling function:

``` r
scaled <- scale_data(split$X_train, split$X_test)
```

Now we can run the XGBoost algorithm with the defined parameters on our
training set:

``` r
trained <- train_xgb_crossvalidated(
  scaled$X_train_scaled,
  split$y_train,
  parameters = params
)
```

Finally we can generate a classification report on our test set:

``` r
report <- generate_classification_report(trained, scaled$X_test_scaled, split$y_test)

knitr::kable(report)
```

|              | precision |    recall |  f1-score |
|:-------------|----------:|----------:|----------:|
| 0.0          | 0.8859915 | 0.9377407 | 0.9111319 |
| 1.0          | 0.6777409 | 0.5204082 | 0.5887446 |
| accuracy     | 0.8538462 | 0.8538462 | 0.8538462 |
| macro avg    | 0.7818662 | 0.7290744 | 0.7499382 |
| weighted avg | 0.8441278 | 0.8538462 | 0.8463238 |
