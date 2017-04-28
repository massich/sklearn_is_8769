from sklearn import linear_model
reg = linear_model.LinearRegression()
X = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
y = list( range(3) )
reg.fit (X, y)
