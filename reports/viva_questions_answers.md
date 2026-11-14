# 🎓 Comprehensive Viva Voce Questions & Answers
## Project: Medical Insurance Price Prediction using Machine Learning
### Designed for College Main / Final Year Project Defense

---

### Q1: What is the main objective of this project?
**Answer:**
The objective of this project is to develop an end-to-end Machine Learning regression system to accurately predict individual medical insurance costs (annual charges) based on demographic, lifestyle, and clinical parameters (age, biological sex, BMI, number of dependents, smoking status, and geographic region). This enables insurance carriers to practice transparent, data-driven actuarial underwriting while helping beneficiaries understand their health risk factors and potential financial savings.

---

### Q2: What dataset did you use, and what are its features?
**Answer:**
We utilized the gold-standard benchmark **Medical Cost Personal Dataset** (originally sourced from the US Census Bureau and healthcare surveys, standard on Kaggle) consisting of 1,338 individual records. The features are:
1. `age`: Primary beneficiary age in years (numerical: 18 to 64).
2. `sex`: Biological gender of policyholder (`male`, `female`).
3. `bmi`: Body Mass Index ($\text{kg/m}^2$), measuring body fat relative to height and weight (ideal range: 18.5 to 24.9).
4. `children`: Number of dependent children covered by health insurance (numerical: 0 to 5).
5. `smoker`: Tobacco smoking habit (`yes` or `no`).
6. `region`: Beneficiary's residential area in the United States (`northeast`, `northwest`, `southeast`, `southwest`).
7. `charges` *(Target Variable)*: Annual medical insurance costs billed to the individual ($).

---

### Q3: Why is this formulated as a Regression problem rather than Classification?
**Answer:**
The target variable (`charges`) is continuous, quantitative, and measured on a dollar scale rather than categorical or discrete class labels. Therefore, regression algorithms (Linear Regression, SVR, Random Forest Regressor, Gradient Boosting Regressor) are mathematically appropriate to predict numeric quantities rather than discrete classes.

---

### Q4: How did you preprocess categorical variables?
**Answer:**
Categorical variables (`sex`, `smoker`, `region`) cannot be processed directly by numerical ML algorithms. We used **One-Hot Encoding** (`OneHotEncoder` from Scikit-Learn) with parameter `drop='first'`.
- `sex`: Encoded into binary 0/1 (`sex_male`).
- `smoker`: Encoded into binary 0/1 (`smoker_yes`).
- `region`: 4 categories encoded into 3 binary indicator columns (`region_northwest`, `region_southeast`, `region_southwest`), where all zeros implicitly represents `northeast`.

---

### Q5: Why did you set `drop='first'` during One-Hot Encoding?
**Answer:**
To avoid the **Dummy Variable Trap**, which causes severe **Multicollinearity**. In linear regression models, if all $k$ categories of a categorical feature are encoded into $k$ binary columns, the sum of those columns equals 1, creating a perfect linear dependency with the intercept column ($\beta_0$). Setting `drop='first'` produces $k-1$ features, eliminating redundant linear combinations and ensuring matrix invertibility $(X^T X)^{-1}$.

---

### Q6: What domain-specific Feature Engineering did you implement?
**Answer:**
Exploratory Data Analysis revealed a pronounced non-linear synergy: smokers with obesity ($\text{BMI} \ge 30$) incur medical charges 4x–5x higher than either condition independently. We engineered:
1. `is_obese`: A binary flag where $\text{BMI} \ge 30.0$.
2. `obese_smoker`: A multiplicative interaction term $(\text{is\_obese} \times \text{smoker\_flag})$.
This interaction feature drastically elevated the $R^2$ accuracy of linear and tree models by directly capturing compounding actuarial risk.

---

### Q7: Why did you scale numerical features using StandardScaler?
**Answer:**
Features like `age` (18–64) and `bmi` (15–53) operate on differing numeric scales. Algorithms relying on distance calculations (like Support Vector Regression) or gradient descent optimization (like regularized linear models) are heavily biased towards features with larger magnitudes without feature scaling. `StandardScaler` standardizes features to zero mean ($\mu = 0$) and unit variance ($\sigma = 1$):
$$z = \frac{x - \mu}{\sigma}$$

---

### Q8: How did you prevent Data Leakage between Train and Test sets?
**Answer:**
We performed an 80/20 train-test split *before* fitting the `StandardScaler` and `OneHotEncoder`. The preprocessor was strictly fitted on `X_train` (`fit_transform`) and subsequently used only to transform `X_test` (`transform`), ensuring no statistical parameters (mean, variance) of the test dataset leaked into model training.

---

### Q9: Which Machine Learning algorithms did you evaluate?
**Answer:**
We systematically benchmarked eight distinct algorithms spanning parametric, linear, distance-based, and ensemble families:
1. **Linear Regression** (Ordinary Least Squares baseline)
2. **Ridge Regression** ($L_2$ regularization)
3. **Lasso Regression** ($L_1$ regularization / sparse feature selection)
4. **Decision Tree Regressor** (CART single tree baseline)
5. **Support Vector Regressor (SVR)** with Radial Basis Function (RBF) kernel
6. **AdaBoost Regressor** (Adaptive sequential boosting)
7. **Random Forest Regressor** (Ensemble bagging of de-correlated decision trees)
8. **Gradient Boosting Regressor** (Gradient descent stage-wise boosting)

---

### Q10: Which model performed the best and why?
**Answer:**
The **Tuned Random Forest / Gradient Boosting Regressor** achieved the highest performance with a Test $R^2$ exceeding $0.86$ and the lowest Root Mean Squared Error ($\approx \$4,400$).
**Why?**
- Medical costs exhibit strong piecewise non-linearities (threshold effects around age 50, BMI 30, and smoker status) that single linear equations cannot model without manual polynomial terms.
- Ensemble tree methods partition feature space hierarchically into high-risk and low-risk subgroups.
- Bagging (Random Forest) reduces variance through bootstrapping and random feature sub-selection at each split, preventing overfitting.

---

### Q11: Explain the evaluation metrics you used: MAE, MSE, RMSE, and $R^2$.
**Answer:**
1. **Mean Absolute Error (MAE):** Average magnitude of absolute errors:
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
   It is in the same units as the target (\$) and robust to extreme outliers.
2. **Mean Squared Error (MSE):** Average of squared differences:
   $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
   Penalizes large prediction errors severely.
3. **Root Mean Squared Error (RMSE):** Square root of MSE:
   $$\text{RMSE} = \sqrt{\text{MSE}}$$
   Brings the metric back into dollar units while still penalizing large deviations.
4. **Coefficient of Determination ($R^2$ Score):** Proportion of variance in target explained by the independent variables:
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
   An $R^2$ of 0.86 means the model accounts for 86% of the variation in insurance costs.

---

### Q12: What is Adjusted $R^2$ and why is it important?
**Answer:**
Ordinary $R^2$ monotonically increases or remains unchanged whenever new features are added to a model, even if those features have zero predictive value. **Adjusted $R^2$** penalizes the inclusion of redundant features based on degrees of freedom:
$$\text{Adjusted } R^2 = 1 - \left[ \frac{(1 - R^2)(n - 1)}{n - p - 1} \right]$$
where $n$ is sample size and $p$ is the number of predictors. It guarantees that our feature engineering truly added genuine predictive power.

---

### Q13: What is K-Fold Cross-Validation?
**Answer:**
In standard train/test split, results can be sensitive to the random partition. In **5-Fold Cross-Validation**, the training set is split into 5 equal subsets (folds). The model is trained on 4 folds and tested on the remaining fold, repeating 5 times such that every fold serves as a test set once. The mean $R^2$ across all 5 folds provides an unbiased, robust estimate of model generalization.

---

### Q14: How did you tune hyperparameters?
**Answer:**
We used **`GridSearchCV`** from Scikit-Learn with 5-fold cross-validation. For Random Forest, we optimized:
- `n_estimators`: Number of decision trees (100, 150, 200).
- `max_depth`: Maximum depth of trees to control complexity (None, 5, 8, 12).
- `min_samples_split`: Minimum samples required to split an internal node (2, 5, 10).
- `min_samples_leaf`: Minimum samples required at leaf nodes (1, 2, 4).

---

### Q15: What is the single most influential feature determining insurance cost?
**Answer:**
Based on Feature Importance analysis (Gini importance from the tree ensemble), **`smoker`** (and the engineered `obese_smoker` interaction) is by far the most dominant factor, contributing over 60% of relative predictive importance, followed by `bmi` and `age`.

---

### Q16: How is the trained model deployed?
**Answer:**
The fitted preprocessing pipeline and the champion tuned model are serialized into binary artifacts (`preprocessor.pkl` and `best_model.pkl`) using Python's `joblib` library. We implemented a dual deployment:
1. **Streamlit Web Application (`app/app.py`):** User-friendly GUI providing single applicant predictions, BMI calculators, lifestyle saving recommendations, interactive EDA, and CSV batch processing.
2. **FastAPI REST API (`app/api.py`):** Production-grade backend API accepting JSON payloads, validated with Pydantic schemas, and exposing `/predict`, `/health`, and `/model/metrics` endpoints with auto-generated Swagger UI.

---

### Q17: What are the limitations and future enhancements?
**Answer:**
**Limitations:**
- Dataset does not include detailed medical history (pre-existing conditions like diabetes, hypertension, cancer history, family genetics).
- Does not capture lifestyle variables like exercise frequency, alcohol consumption, or occupation hazard.
**Future Enhancements:**
- Integrating electronic health record (EHR) data.
- Implementing Explainable AI (SHAP / LIME values) for individual local prediction explainability.
- Deploying containerized microservices using Docker on AWS / GCP / Azure.
