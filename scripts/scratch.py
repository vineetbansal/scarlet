from scarlet.numeric import np

x = np.arange(100).reshape((10, 5, 2))
print(type(x))
print(x.shape)
print(x.dtype)

x = x[:5, :, :]
print(type(x))
print(x.shape)
print(x.dtype)

x = np.array((1, 2, 3))
print(type(x))
print(x.shape)
print(x.dtype)

x = np.array([[1, 2, 3], [4, 5, 6]])
print(type(x))
print(x.shape)
print(x.dtype)

x = np.arange(100).reshape((10, 5, 2)).astype('float')
print(type(x))
print(x.shape)
print(x.dtype)

x = np.arange(100).reshape((10, 5, 2)).astype('complex')
print(type(x))
print(x.shape)
print(x.dtype)