import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import matplotlib.pyplot as plt
# StandardScaler, whose main parameters are with_mean and with_std, both Booleans, indicating whether the algorithm should zero-center and whether it should divide by the standard deviations. The default values are both True.
# MinMaxScaler, whose main parameter is feature_range, which requires a tuple or list of two elements (a, b) so that a < b. The default value is (0, 1).
# RobustScaler, which is mainly based on the parameter quantile_range. The default is (25, 75) corresponding to the IQR. In a similar way to StandardScaler, the class accepts the parameters with_centering and with_scaling, that selectively activate/deactivate each of the two functions.

n_sample  = 200

mean_ = [1.0,1.0]
coxv_ = [[2.0, 0.0], [0.0, 0.8]]

X_data = np.random.multivariate_normal(mean=mean_, cov=coxv_, size=n_sample)

print(X_data.shape)

# scaler = StandardScaler()
# x_scaler = scaler.fit(X_data)
# print(x_scaler)

# X_rss = RobustScaler(quantile_range=(10, 90))
# X_rss.fit(X_data)

# X_min = MinMaxScaler(feature_range=(1, 80))
# X_min.fit(X_data)

# from sklearn.preprocessing import Normalizer

# normalizer = Normalizer()
# normalised = normalizer.fit_transform(X_data)

# plot the normalised data against the original x
#plt.scatter(x= X_data, y=normalised)

# zero centered 

from sklearn.model_selection import train_test_split

x_train, x_test = train_test_split(X_data, train_size=0.7,random_state=1000)
print(len(x_train))
print(len(x_test))

def zeroCenter(data_X):
    return np.mean(data_X, axis=0)

result = zeroCenter(X_data)
print(result)


# plt.plot(normalised)
# plt.xlabel('Original x data')
# plt.ylabel('Normalized X data')
# plt.legend()
# plt.show()
