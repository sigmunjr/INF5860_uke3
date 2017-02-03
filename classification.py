import numpy as np


class NearestNeighborClassification(object):
  def fit(self, x, y):
    self.x, self.y = x, y

  def predict(self, x, batch_size=32):
    x1 = self.x[np.newaxis]# - x[:, np.newaxis]
    x2 = x[:, np.newaxis]
    x_dims = tuple(range(len(x2.shape))[2:])
    y_ = []
    print "_"*((x.shape[0]/batch_size)*2 + 1)
    for i in range(0, x.shape[0], batch_size):
      end = i+batch_size if (i+batch_size)<x2.shape[0] else None
      ind = (x1-x2[i:end]).sum(x_dims).argmin(1)
      y_.append(self.y[ind])
      print "*",

    print "\nFINSIHED"
    return np.concatenate(y_)

if __name__ == '__main__':
  from cifar import load_cifar_file

  images, labels = load_cifar_file('data_batch_1')
  print images.shape, labels.shape

  from classification import NearestNeighborClassification

  clf = NearestNeighborClassification()
  clf.fit(images, labels)
  y_ = clf.predict(images[:200])