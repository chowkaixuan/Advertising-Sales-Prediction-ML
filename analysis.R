# ============================================================
# Advertising Sales Prediction (kNN vs Linear Regression)
# Author: Chow Kai Xuan
# ============================================================


# -----------------------------
# 1) Load Required Libraries
# -----------------------------
# Install once if needed:
# install.packages(c("datarium", "ggplot2", "psych", "corrplot", "caret"))

library(ggplot2)
library(psych)
library(corrplot)
library(caret)


# -----------------------------
# 2) Create images folder (if missing)
# -----------------------------
if (!dir.exists("images")) dir.create("images")


# -----------------------------
# 3) Load Dataset
# -----------------------------
data(marketing, package = "datarium")

summary(marketing)


# -----------------------------
# 4) Exploratory Data Analysis
# -----------------------------
cat("\n===== EDA: Summary =====\n")
print(summary(marketing))

# Histogram of sales
ggplot(marketing, aes(sales)) + geom_histogram() + labs(title = "Distribution of Sales")

# Scatterplot matrix + correlations
pairs.panels(marketing, 
             method = "pearson", 
             hist.col = "steelblue",
             pch = 21,
             density = TRUE,
             ellipses = FALSE
)

# Correlation plot
corrplot(cor(marketing), type="upper",
         method="color",
         addCoef.col = "black",
         number.cex = 0.6)


# -----------------------------
# 4) Train-Test Split (80/20)
# -----------------------------
set.seed(100) #reproducible split
training_idx <- sample(1:nrow(marketing), size = nrow(marketing)*0.8)
train_data <- marketing[training_idx, ]
test_data <- marketing[-training_idx, ]

cat("\n===== Data Split =====\n")
cat("Train rows:", nrow(train_data), "\n")
cat("Test rows :", nrow(test_data), "\n")


# -----------------------------
# 6) Model 1 — kNN Regression (caret)
# -----------------------------
set.seed(101)

# 4-fold CV: train has ~160 rows => ~40 rows per fold (reasonable fold size)
knn_model <- train(
  sales~., data = train_data, 
  method = "knn",
  trControl = trainControl("cv", number = 4),
  preProcess = c("center", "scale"),
  tuneLength = 10
)

cat("\n===== kNN Model =====\n")
cat("Best tuning parameter (k):\n")
print(knn_model$bestTune)

# Save RMSE vs k plot
png("images/knn-rmse-vs-k.png", width = 900, height = 600)
# Plot model error RMSE vs different values of k
plot(knn_model)
dev.off()

# Predict on test set + RMSE
knn_pred <- predict(knn_model, test_data)
knn_rmse <- RMSE(knn_pred, test_data$sales)

cat("kNN Test RMSE:", knn_rmse, "\n")

# Save Predicted vs Actual (kNN)
png("images/knn-pred-vs-actual.png", width = 900, height = 600)
plot(test_data$sales, knn_pred,main="kNN: Predicted vs Actual",
     xlab = "Actual Sales",
     ylab = "Predicted Sales"
) 
abline(0,1, col="red")
# Most data points are close to the 45 degree line, implying a very accurate prediction
dev.off()

# -----------------------------
# 7) Model 2 — Linear Regression (Baseline)
# -----------------------------
cat("\n===== Linear Regression (Baseline) =====\n")

lmodel<- lm(sales~., data = train_data) 
summary(lmodel)

lmodel_predict <- predict(lmodel, test_data)
lmodel_rmse <- RMSE(lmodel_predict, test_data$sales)

cat("Baseline LM Test RMSE:", lmodel_rmse, "\n")

# Save Predicted vs Actual (Baseline LM)
png("images/lmodel-baseline-pred-vs-actual.png", width = 900, height = 600)
plot(
  test_data$sales, lmodel_predict,
  main = "Baseline LM: Predicted vs Actual",
  xlab = "Actual Sales",
  ylab = "Predicted Sales"
)
abline(0,1, col="red")
dev.off()

# Baseline residual diagnostics
png("images/lm-baseline-residuals.png", width = 1000, height = 800)
par(mfrow = c(2, 2))
plot(lmodel)
dev.off()
par(mfrow = c(1, 1))



# -----------------------------
# 8) Residual Diagnostics + Model Improvement
# -----------------------------
# We check residuals of the baseline model:
# - If residual plots show curvature/nonlinearity, add polynomial terms
# - If interaction is plausible, add interaction term(s)

# Improved model using polynomial terms and interaction:
# Based on EDA, youtube and facebook tend to be more strongly related to sales.
# We add youtube^2, facebook^2, and youtube*facebook.

lmodel_best <- lm(sales ~ youtube + facebook + newspaper + 
                    I(youtube^2) + 
                    I(youtube * facebook) + 
                    I(facebook^2), 
                  data = train_data
)
cat("\n===== Linear Regression (Enhanced) =====\n")
summary(lmodel_best)

lmodel_best_pred <-predict(lmodel_best, test_data)
lmodel_best_rmse <- RMSE(lmodel_best_pred, test_data$sales)
cat("Enhanced LM Test RMSE:", lmodel_best_rmse, "\n")


# -----------------------------
# 9) Save Key Portfolio Plots (Enhanced LM)
# -----------------------------

# Enhanced model: predicted vs actual
png("images/final-model-pred-vs-actual.png", width = 900, height = 600)
plot(
  test_data$sales, lmodel_best_pred,
  main = "Enhanced LM: Predicted vs Actual",
  xlab = "Actual Sales",
  ylab = "Predicted Sales"
)
abline(0, 1, col = "red")
dev.off()

# Enhanced model: residual diagnostics
png("images/final-model-residuals.png", width = 1000, height = 800)
par(mfrow = c(2, 2))
plot(lmodel_best)
dev.off()
par(mfrow = c(1, 1))

# -----------------------------
# 10) Model Comparison Summary
# -----------------------------
cat("\n===== Model Comparison (Test RMSE) =====\n")

results <- data.frame(
  Model = c("kNN", "Linear Regression (Baseline)", "Linear Regression (Enhanced)"),
  RMSE  = c(knn_rmse, lmodel_rmse, lmodel_best_rmse)
)

print(results)

cat("\nConclusion:\n")
cat("- Choose the model with the lowest test RMSE.\n")
cat("- For interpretability, linear regression is easier to explain than kNN.\n")
