import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import VotingRegressor

# Load Data:
X, y = load_diabetes(return_X_y=True)

# Model Specification
base_lm  = LinearRegression()
ridge_lm = Ridge(alpha = 0.2)
lasso_lm = Lasso(alpha = 0.2)
voting_m = VotingRegressor([
  ("base", base_lm),
  ("l1", lasso_lm), 
  ("l2", ridge_lm)])

# Fit models
base_lm.fit(X, y)
ridge_lm.fit(X, y)
lasso_lm.fit(X, y)
voting_m.fit(X, y)

# Predict first 8 observations
xt = X[:8]

base_pred = base_lm.predict(xt)
ridge_pred = ridge_lm.predict(xt)
lasso_pred = lasso_lm.predict(xt)
voting_pred = voting_m.predict(xt)

#Plot figure
plt.figure()
plt.plot(base_pred, "gd", label="Base prediction")
plt.plot(ridge_pred, "b^", label="Ridge prediction")
plt.plot(lasso_pred, "ys", label="LASSO prediction")
plt.plot(voting_pred, "r*", ms=10, label="VotingRegressor")

plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
plt.ylabel("predicted")
plt.xlabel("training samples")
plt.legend(loc="best")
plt.title("Regressor predictions and their average")

plt.show()