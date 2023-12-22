import pandas as pd

relu = pd.read_csv("mnist_relu_cv.csv")
cosine = pd.read_csv("mnist_cosine_cv.csv")

relu_mean_accuracy = relu["accuracy"].mean()
cosine_mean_accuracy = cosine["accuracy"].mean()
relu_sd = relu["accuracy"].std()
cosine_sd = cosine["accuracy"].std()

print("relu mean accuracy: ", relu_mean_accuracy)
print("relu sd: ", relu_sd)
print("cosine mean accuracy: ", cosine_mean_accuracy)
print("cosine sd: ", cosine_sd)

# Check normality assumptions with a normal quantile plot
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot

# q-q plot
qqplot(relu["accuracy"], line='s')
pyplot.show()

qqplot(cosine["accuracy"], line='s')
pyplot.show()

# Check for significant difference
from scipy.stats import ttest_ind

# Student's t-test
stat, p = ttest_ind(cosine["accuracy"], relu["accuracy"])
print('t=%.3f, p=%.10f' % (stat, p))
print('t-test p-value: %.10f' % p)
