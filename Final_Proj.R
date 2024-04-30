library(e1071)
library(glmnet)
library(class)
library(tree)
library(ggcorrplot)
set.seed(222)

#### Data Set Up ####

# Read in full data frame
data = read.csv('Desktop/STA4241/Final_Proj_Data/Variogram.csv')
cols = ncol(data)

# Split data into categories
sgs = data[1:100,]
dcgan = data[101:200,]
diffusion = data[201:300,]

# Change diffusion indicator from 2 to 1 for binary comparison with sgs
diffusion[,ncol(diffusion)] = 1

# concatenate data frames
sgs_dcgan = rbind(sgs, dcgan)
sgs_diffusion = rbind(sgs, diffusion)

# shuffle rows
sgs_dcgan = sgs_dcgan[sample(1:nrow(sgs_dcgan)),]
sgs_diffusion = sgs_diffusion[sample(1:nrow(sgs_diffusion)),]

# train / test split (75% / 25%)
rows = -1
if (nrow(sgs_dcgan) == nrow(sgs_diffusion)){
  rows = nrow(sgs_dcgan)
} else{
  errorCondition("Not equal num of rows")
}

smp_size = floor(0.75 * rows)
train_ind = sample(seq_len(rows), size = smp_size)

sgs_dcgan_train = sgs_dcgan[train_ind,]
sgs_dcgan_test = sgs_dcgan[-train_ind,]

sgs_diffusion_train = sgs_diffusion[train_ind,]
sgs_diffusion_test = sgs_diffusion[-train_ind,]



#### Fit all the full models ####

# (1) Logistic regression w/ linear terms
dcgan_lin_logit = glm(Y~.,family=binomial(link=logit),data=sgs_dcgan_train)
diffusion_lin_logit = glm(Y~.,family=binomial(link=logit),data=sgs_diffusion_train)

# (2) Logistic regression w/ squared terms
formula = as.formula(paste0("Y~",paste0("s(Lag_",1:30,", df = 2)",collapse="+")))
dcgan_sq_logit=glm(formula,family=binomial(link=logit),data=sgs_dcgan_train)
diffusion_sq_logit=glm(formula,family=binomial(link=logit),data=sgs_diffusion_train)

# (3) Probit Regression
dcgan_probit = glm(Y~., family=binomial(link=probit), data=sgs_dcgan_train)
diffusion_probit = glm(Y~., family=binomial(link=probit), data=sgs_diffusion_train)

# (4) Linear discriminant analysis
dcgan_LDA = lda(Y~., data=sgs_dcgan_train)
# diffusion_LDA = lda(Y~., data=sgs_diffusion_train)

# (5) Quadratic discriminant analysis
dcgan_QDA = qda(Y~., data=sgs_dcgan_train)
diffusion_QDA = qda(Y~., data=sgs_diffusion_train)

# (6) SVM with Linear Kernel
dcgaan_svm_lin = tune(e1071::svm, as.factor(Y)~., data=sgs_dcgan_train, kernel="linear")
diffusion_svm_lin = tune(e1071::svm, as.factor(Y)~., data=sgs_diffusion_train, kernel="linear")

# (7) SVM with Radial Kernel
dcgaan_svm_rad = tune(e1071::svm, as.factor(Y)~., data=sgs_dcgan_train, kernel="radial")
diffusion_svm_rad = tune(e1071::svm, as.factor(Y)~., data=sgs_diffusion_train, kernel="radial")

# (8) Regression Tree w/ pruning and CV depth 
dc_tree = tree(as.factor(Y)~., data=sgs_dcgan_train)
dc_cv_tree = cv.tree(dc_tree, FUN = prune.misclass)
dc_depth = dc_cv_tree$size[which.min(dc_cv_tree$dev)]
dc_prune_tree = prune.misclass(dc_tree, best=dc_depth)
df_tree = tree(as.factor(Y)~., data=sgs_diffusion_train)
df_cv_tree = cv.tree(df_tree, FUN = prune.misclass)
df_depth = df_cv_tree$size[which.min(df_cv_tree$dev)]
df_prune_tree = prune.misclass(df_tree, best=df_depth)



#### Predict using full models ####

errorMat = matrix(NA, 2, 8)

dc_test = sgs_dcgan_test
df_test = sgs_diffusion_test

# (1) Logistic regression w/ linear terms
dc_logit_pred = predict(dcgan_lin_logit, dc_test[,-cols], type="response")
errorMat[1,1] = sum((dc_logit_pred>=0.5) != dc_test[,cols])/nrow(dc_test)
df_logit_pred = predict(diffusion_lin_logit, df_test[,-cols], type="response")
errorMat[2,1] = sum((df_logit_pred>=0.5) != df_test[,cols])/nrow(df_test)

# (2) Logistic regression w/ squared terms
dc_sq_logit_pred = predict(dcgan_sq_logit, dc_test[,-cols], type="response")
errorMat[1,2] = sum((dc_sq_logit_pred>=0.5) != dc_test[,cols])/nrow(dc_test)
df_sq_logit_pred = predict(diffusion_sq_logit, df_test[,-cols], type="response")
errorMat[2,2] = sum((df_sq_logit_pred>=0.5) != df_test[,cols])/nrow(df_test)

# (3) Probit Regression
dc_probit_pred = predict(dcgan_probit, dc_test[,-cols], type="response")
errorMat[1,3] = sum((dc_probit_pred>=0.5) != dc_test[,cols])/nrow(dc_test)
df_probit_pred = predict(diffusion_probit, df_test[,-cols], type="response")
errorMat[2,3] = sum((df_probit_pred>=0.5) != df_test[,cols])/nrow(df_test)

# (4) Linear discriminant analysis
dc_lda_pred = predict(dcgan_LDA, dc_test[,-cols], type="response")
errorMat[1,4] = sum((as.numeric(dc_lda_pred$class)-1) != dc_test[,cols])/nrow(dc_test)
# Error for diffusion lda (covariance matrix to be singular)

# (5) Quadratic discriminant analysis
dc_qda_pred = predict(dcgan_QDA, dc_test[,-cols], type="response")
errorMat[1,5] = sum((as.numeric(dc_qda_pred$class)-1) != dc_test[,cols])/nrow(dc_test)
df_qda_pred = predict(diffusion_QDA, df_test[,-cols], type="response")
errorMat[2,5] = sum((as.numeric(df_qda_pred$class)-1) != df_test[,cols])/nrow(df_test)

# (6) SVM with Linear Kernel
dc_svm_lin_pred = predict(dcgaan_svm_lin$best.model, dc_test[,-cols], type="response")
errorMat[1,6] = sum(dc_svm_lin_pred != dc_test[,cols])/nrow(dc_test)
df_svm_lin_pred = predict(diffusion_svm_lin$best.model, df_test[,-cols], type="response")
errorMat[2,6] = sum(df_svm_lin_pred != df_test[,cols])/nrow(df_test)

# (7) SVM with Radial Kernel
dc_svm_rad_pred = predict(dcgaan_svm_rad$best.model, dc_test[,-cols], type="response")
errorMat[1,7] = sum(dc_svm_rad_pred != dc_test[,cols])/nrow(dc_test)
df_svm_rad_pred = predict(diffusion_svm_rad$best.model, df_test[,-cols], type="response")
errorMat[2,7] = sum(df_svm_rad_pred != df_test[,cols])/nrow(df_test)

# (8) Regression Tree w/ pruning and CV depth 
dc_tree_pred = predict(dc_prune_tree, dc_test[,-cols], type="class")
errorMat[1,8] = sum(dc_tree_pred != dc_test[,cols])/length(dc_test)
df_tree_pred = predict(df_prune_tree, df_test[,-cols], type="class")
errorMat[2,8] = sum(df_tree_pred != df_test[,cols])/length(df_test)


# Error's are very low => Easy to classify sgs vs [GAN/Diffusion]
# Not good for our purposes to compare performance and attain ~50% error rate for at least one DNN
# hmmmmmmm........
# the variograms look very similar graphically even when overlaid
# sad D..:
errorMat



#### Error Handling ####

# Lets look at the data ...
head(sgs)
head(dcgan)
head(diffusion)

# Notice that the lag covariates appear to be shifted by a constant offset 
# => lead to trivial classification of <, > 
plot(dc_prune_tree)
text(dc_prune_tree, pretty = 0)

plot(df_prune_tree)
text(df_prune_tree, pretty = 0)

# We are more concerned with the relationshop than the scale of covariates  => normalize
# EEEEE i think this will work :DD

# Tests to see difference in scale
sgs_means = colMeans(sgs[,-cols])
gan_means = colMeans(dcgan[,-cols])
diffusion_means = colMeans(diffusion[,-cols])

# SSE for difference in covariate means
gan_sse = sum((gan_means - sgs_means)^2); gan_sse
diffusion_sse = sum((diffusion_means - sgs_means)^2); diffusion_sse

# directional difference in covariate means
gan_dir_err = sum((gan_means > sgs_means))/cols; gan_dir_err
diffusion_dir_err = sum((diffusion_means > sgs_means))/cols; diffusion_dir_err



#### Fix Result => Normalize Covariates ####

dcgan_norm_train = sgs_dcgan_train
dcgan_norm_test = sgs_dcgan_test

dcgan_norm_train[,-cols] = t(scale(t(dcgan_norm_train[,-cols])))
dcgan_norm_test[,-cols] = t(scale(t(dcgan_norm_test[,-cols])))

diffusion_norm_train = sgs_diffusion_train
diffusion_norm_test = sgs_diffusion_test

diffusion_norm_train[,-cols] = t(scale(t(diffusion_norm_train[,-cols])))
diffusion_norm_test[,-cols] = t(scale(t(diffusion_norm_test[,-cols])))



#### Fit all the full models on normalized covariates ####

# (1) Logistic regression w/ linear terms
dcgan_lin_logit = glm(Y~.,family=binomial(link=logit),data=dcgan_norm_train)
diffusion_lin_logit = glm(Y~.,family=binomial(link=logit),data=diffusion_norm_train)

# (2) Logistic regression w/ squared terms
formula = as.formula(paste0("Y~",paste0("s(Lag_",1:30,", df = 2)",collapse="+")))
dcgan_sq_logit=glm(formula,family=binomial(link=logit),data=dcgan_norm_train)
diffusion_sq_logit=glm(formula,family=binomial(link=logit),data=diffusion_norm_train)

# (3) Probit Regression
dcgan_probit = glm(Y~., family=binomial(link=probit), data=dcgan_norm_train)
diffusion_probit = glm(Y~., family=binomial(link=probit), data=diffusion_norm_train)

# (4) Linear discriminant analysis
## Warning: collinear variables
dcgan_LDA = lda(Y~., data=dcgan_norm_train)
diffusion_LDA = lda(Y~., data=diffusion_norm_train)

# (5) Quadratic discriminant analysis
## Err: rank deficiency (fix later with PCA)
#dcgan_QDA = qda(Y~., data=dcgan_norm_train) 
#diffusion_QDA = qda(Y~., data=diffusion_norm_train) 

# (6) SVM with Linear Kernel
dcgaan_svm_lin = tune(e1071::svm, as.factor(Y)~., data=dcgan_norm_train, kernel="linear")
diffusion_svm_lin = tune(e1071::svm, as.factor(Y)~., data=diffusion_norm_train, kernel="linear")

# (7) SVM with Radial Kernel
dcgaan_svm_rad = tune(e1071::svm, as.factor(Y)~., data=dcgan_norm_train, kernel="radial")
diffusion_svm_rad = tune(e1071::svm, as.factor(Y)~., data=diffusion_norm_train, kernel="radial")

# (8) Regression Tree w/ pruning and CV depth 
dc_tree = tree(as.factor(Y)~., data=dcgan_norm_train)
dc_cv_tree = cv.tree(dc_tree, FUN = prune.misclass)
dc_depth = dc_cv_tree$size[which.min(dc_cv_tree$dev)]
dc_prune_tree = prune.misclass(dc_tree, best=dc_depth)
df_tree = tree(as.factor(Y)~., data=diffusion_norm_train)
df_cv_tree = cv.tree(df_tree, FUN = prune.misclass)
df_depth = df_cv_tree$size[which.min(df_cv_tree$dev)]
if(df_depth == 1){df_depth = 2}
df_prune_tree = prune.misclass(df_tree, best=df_depth)



#### Predict using full models with normalized covariates ####

errorMatNorm = matrix(NA, 2, 8)

dc_test = dcgan_norm_test
df_test = diffusion_norm_test

# (1) Logistic regression w/ linear terms
dc_logit_pred = predict(dcgan_lin_logit, dc_test[,-cols], type="response")
errorMatNorm[1,1] = sum((dc_logit_pred>=0.5) != dc_test[,cols])/nrow(dc_test)
df_logit_pred = predict(diffusion_lin_logit, df_test[,-cols], type="response")
errorMatNorm[2,1] = sum((df_logit_pred>=0.5) != df_test[,cols])/nrow(df_test)

# (2) Logistic regression w/ squared terms
dc_sq_logit_pred = predict(dcgan_sq_logit, dc_test[,-cols], type="response")
errorMatNorm[1,2] = sum((dc_sq_logit_pred>=0.5) != dc_test[,cols])/nrow(dc_test)
df_sq_logit_pred = predict(diffusion_sq_logit, df_test[,-cols], type="response")
errorMatNorm[2,2] = sum((df_sq_logit_pred>=0.5) != df_test[,cols])/nrow(df_test)

# (3) Probit Regression
dc_probit_pred = predict(dcgan_probit, dc_test[,-cols], type="response")
errorMatNorm[1,3] = sum((dc_probit_pred>=0.5) != dc_test[,cols])/nrow(dc_test)
df_probit_pred = predict(diffusion_probit, df_test[,-cols], type="response")
errorMatNorm[2,3] = sum((df_probit_pred>=0.5) != df_test[,cols])/nrow(df_test)

# (4) Linear discriminant analysis
dc_lda_pred = predict(dcgan_LDA, dc_test[,-cols], type="response")
errorMatNorm[1,4] = sum((as.numeric(dc_lda_pred$class)-1) != dc_test[,cols])/nrow(dc_test)
df_lda_pred = predict(diffusion_LDA, df_test[,-cols], type="response")
errorMatNorm[2,4] = sum((as.numeric(df_lda_pred$class)-1) != df_test[,cols])/nrow(df_test)

# (5) Quadratic discriminant analysis
# Err: rank deficiency (fix later with PCA)

# (6) SVM with Linear Kernel
dc_svm_lin_pred = predict(dcgaan_svm_lin$best.model, dc_test[,-cols], type="response")
errorMatNorm[1,6] = sum(dc_svm_lin_pred != dc_test[,cols])/nrow(dc_test)
df_svm_lin_pred = predict(diffusion_svm_lin$best.model, df_test[,-cols], type="response")
errorMatNorm[2,6] = sum(df_svm_lin_pred != df_test[,cols])/nrow(df_test)

# (7) SVM with Radial Kernel
dc_svm_rad_pred = predict(dcgaan_svm_rad$best.model, dc_test[,-cols], type="response")
errorMatNorm[1,7] = sum(dc_svm_rad_pred != dc_test[,cols])/nrow(dc_test)
df_svm_rad_pred = predict(diffusion_svm_rad$best.model, df_test[,-cols], type="response")
errorMatNorm[2,7] = sum(df_svm_rad_pred != df_test[,cols])/nrow(df_test)

# (8) Regression Tree w/ pruning and CV depth 
dc_tree_pred = predict(dc_prune_tree, dc_test[,-cols], type="class")
errorMatNorm[1,8] = sum(dc_tree_pred != dc_test[,cols])/nrow(dc_test)
df_tree_pred = predict(df_prune_tree, df_test[,-cols], type="class")
errorMatNorm[2,8] = sum(df_tree_pred != df_test[,cols])/nrow(df_test)

errorMatNorm



#### Fit all the reduced models ####

# Separate covariates from outcome 
dc_x = as.matrix(dcgan_norm_train[,-cols])
dc_y = as.numeric(dcgan_norm_train[,cols])
dc_x_test = as.matrix(dcgan_norm_test[,-cols])
dc_y_test = as.numeric(dcgan_norm_test[,cols])

cor_matrix = cor(dc_x)
ggcorrplot(cor_matrix)

df_x = as.matrix(diffusion_norm_train[,-cols])
df_y = as.numeric(diffusion_norm_train[,cols])
df_x_test = as.matrix(diffusion_norm_test[,-cols])
df_y_test = as.numeric(diffusion_norm_test[,cols])

cor_matrix = cor(df_x)
ggcorrplot(cor_matrix)

# Get PCs that explain 95% Variability
dc_comp = prcomp(dc_x)
dc_above_95 = cumsum(dc_comp$sdev^2/sum(dc_comp$sdev^2)) > 0.95
dc_num_pc = min(which(dc_above_95 == TRUE)); dc_num_pc
dc_pc_train = data.frame(cbind(prcomp(dc_x, rank = dc_num_pc)$x, dc_y))
dc_pc_test = data.frame(cbind(prcomp(dc_x_test, rank = dc_num_pc)$x, dc_y_test))

df_comp = prcomp(df_x)
df_above_95 = cumsum(df_comp$sdev^2/sum(df_comp$sdev^2)) > 0.95
df_num_pc = min(which(df_above_95 == TRUE)); df_num_pc
df_pc_train = data.frame(cbind(prcomp(df_x, rank = df_num_pc)$x, df_y))
df_pc_test = data.frame(cbind(prcomp(df_x_test, rank = df_num_pc)$x, df_y_test))


# (1) Logistic Regression with Lasso Penalization (w/ Cross Validation)
dc_lasso_cv = cv.glmnet(dc_x, dc_y, alpha = 1, family = "binomial")
dc_lasso = glmnet(dc_x, dc_y, alpha = 1, family = "binomial",lambda = dc_lasso_cv$lambda.min)
df_lasso_cv = cv.glmnet(df_x, df_y, alpha = 1, family = "binomial")
df_lasso = glmnet(df_x, df_y, alpha = 1, family = "binomial",lambda = df_lasso_cv$lambda.min)

# (2) Logistic Regression with Ridge Penalization (w/ Cross Validation)
dc_ridge_cv = cv.glmnet(dc_x, dc_y, alpha = 0, family = "binomial")
dc_ridge = glmnet(dc_x, dc_y, alpha = 0, family = "binomial",lambda = dc_ridge_cv$lambda.min)
df_ridge_cv = cv.glmnet(df_x, df_y, alpha = 0, family = "binomial")
df_ridge = glmnet(df_x, df_y, alpha = 0, family = "binomial",lambda = df_ridge_cv$lambda.min)

# (3) Logistic Regression (w/ 95% Variability explained by PCs)
dc_logit_pc = glm(dc_y~., family=binomial(link=logit), data=dc_pc_train)
df_logit_pc = glm(df_y~., family=binomial(link=logit), data=df_pc_train)

# (4) Linear discriminant analysis (w/ 95% Variability explained by PCs)
dc_LDA_pc = lda(dc_y~., data=dc_pc_train)
df_LDA_pc = lda(df_y~., data=df_pc_train)

# (5) Quadratic discriminant analysis (w/ 95% Variability explained by PCs)
dc_QDA_pc = qda(dc_y~., data=dc_pc_train)
df_QDA_pc = qda(df_y~., data=df_pc_train)

# (6) KNN (w/ 95% Variability explained by PCs)
dc_knn_pc = knn(train = dc_pc, test = dc_pc_test[,-1], cl = dc_y, k = 10)
df_knn_pc = knn(train = df_pc, test = df_pc_test[,-1], cl = df_y, k = 10)

# (7) Regression Tree w/ pruning and CV depth (w/ 95% Variability explained by PCs)
dc_fit = tree(as.factor(dc_y)~. ,data=dc_pc_train)
dc_cv_fit = cv.tree(dc_fit, FUN = prune.misclass)
dc_depth = dc_cv_fit$size[which.min(dc_cv_fit$dev)]
dc_prune_tree = prune.misclass(dc_fit, best=dc_depth)
df_fit = tree(as.factor(df_y)~. ,data=df_pc_train)
df_cv_fit = cv.tree(df_fit, FUN = prune.misclass)
df_depth = df_cv_fit$size[which.min(df_cv_fit$dev)]
df_prune_tree = prune.misclass(df_fit, best=df_depth) 



#### Predict using reduced models ####

errorMatReduced = matrix(NA, 2, 7)

# (1) Logistic Regression with Lasso Penalization (w/ Cross Validation)
dc_lasso_pred = predict(dc_lasso, dc_x_test, type="response")
errorMatReduced[1,1] = sum((dc_lasso_pred>=0.5) != dc_y_test)/length(dc_y_test)
df_lasso_pred = predict(df_lasso, df_x_test, type="response")
errorMatReduced[2,1] = sum((df_lasso_pred>=0.5) != df_y_test)/length(df_y_test)

# (2) Logistic Regression with Ridge Penalization (w/ Cross Validation)
dc_ridge_pred = predict(dc_ridge, dc_x_test, type="response")
errorMatReduced[1,2] = sum((dc_ridge_pred>=0.5) != dc_y_test)/length(dc_y_test)
df_ridge_pred = predict(df_ridge, df_x_test, type="response")
errorMatReduced[2,2] = sum((df_ridge_pred>=0.5) != df_y_test)/length(df_y_test)

# (3) Logistic Regression (w/ 95% Variability explained by PCs)
dc_logit_pc_pred = predict(dc_logit_pc, dc_pc_test)
errorMatReduced[1,3] = sum((dc_logit_pc_pred>=0.5) != dc_y_test)/length(dc_y_test)
df_logit_pc_pred = predict(df_logit_pc, df_pc_test, type="response")
errorMatReduced[2,3] = sum((df_logit_pc_pred>=0.5) != df_y_test)/length(df_y_test)

# (4) Linear discriminant analysis (w/ 95% Variability explained by PCs)
dc_lda_pc_pred = predict(dc_LDA_pc, dc_pc_test, type="response")
errorMatReduced[1,4] = sum((as.numeric(dc_lda_pc_pred$class)-1) != dc_y_test)/length(dc_y_test)
df_lda_pc_pred = predict(df_LDA_pc, df_pc_test, type="response")
errorMatReduced[2,4] = sum((as.numeric(df_lda_pc_pred$class)-1) != df_y_test)/length(df_y_test)

# (5) Quadratic discriminant analysis (w/ 95% Variability explained by PCs)
dc_qda_pc_pred = predict(dc_QDA_pc, dc_pc_test, type="response")
errorMatReduced[1,5] = sum((as.numeric(dc_qda_pc_pred$class)-1) != dc_y_test)/length(dc_y_test)
df_qda_pc_pred = predict(df_QDA_pc, df_pc_test, type="response")
errorMatReduced[2,5] = sum((as.numeric(df_qda_pc_pred$class)-1) != df_y_test)/length(df_y_test)

# (6) KNN (w/ 95% Variability explained by PCs)
errorMatReduced[1,6] = sum(dc_knn_pc != dc_y_test)/length(dc_y_test)
errorMatReduced[2,6] = sum(df_knn_pc != df_y_test)/length(df_y_test)

# (7) Regression Tree w/ pruning and CV depth (w/ 95% Variability explained by PCs)
dc_tree_pred = predict(dc_prune_tree, dc_pc_test, type="class")
errorMatReduced[1,7] = sum(dc_tree_pred != dc_y_test)/length(dc_y_test)
df_tree_pred = predict(df_prune_tree, df_pc_test, type="class")
errorMatReduced[2,7] = sum(df_tree_pred != df_y_test)/length(df_y_test)

errorMatReduced
