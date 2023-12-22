import random
import math
import numpy as np

def surface_data(samples=50, sigma=0):

    r = np.linspace(0, 1.0, samples)
    a = np.linspace(0, 2 * np.pi, samples, endpoint = False)
    a = np.repeat(a[..., np.newaxis], samples, axis = 1) 

    x = np.append(0, (r * np.cos(a))).reshape(-1, 1) 
    y = np.append(0, (r * np.sin(a))).reshape(-1, 1)   
    z = np.sin(x * y).reshape(-1, 1)
    e = np.random.normal(0, sigma, len(z)).reshape(-1, 1)

    return x, y, z + e

def plane_data(samples=50, sigma=0):

    a = np.linspace(0, 1.0, samples)
    b = np.linspace(0, 1.0, samples)
    x, y = np.meshgrid(a, b)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = x - y
    e = np.random.normal(0, sigma, len(z)).reshape(-1, 1)

    return x, y, z + e

def spiral_data(samples=100, classes=3):

    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        r = np.linspace(0.0, 1, samples)
        t = np.linspace(class_number*4, (class_number+1)*4,
                        samples) + np.random.randn(samples)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number

    return X, y

def wave_data(samples=1000):

    features = np.zeros((samples, 2))
    target = np.zeros(samples).astype(int)
    for i in range(samples):
        x = (random.random() - 0.5) * 8
        y = (random.random() - 0.5) * 8
        features[i] = np.array([x, y])

        class_no = 0
        if math.cos(x * y) + 2 * (random.random() - 0.5) > math.sin(x * y):
            class_no = 1
        target[i] = int(class_no)

    return features, target

def sine_data(samples=1000, sigma=0):

    X = np.arange(samples).reshape(-1, 1) / samples
    y = np.sin(2 * np.pi * X).reshape(-1, 1)
    e = np.random.normal(0, sigma, samples).reshape(-1, 1)

    return X, y + e