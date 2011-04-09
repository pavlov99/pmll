import numpy as np

class ObjectFeatureMatrix(object):
    """
    Class for representation of objects-by-features matrix of features.
    """
    def __init__(self, matrix=None):
        self.objects = None
        self.nobjects = 0 
        self.nfeatures = None

        if matrix is not None:
            self.add(matrix)

    def add(self, matrix):
        matrix = np.asmatrix(matrix)
        if self.nfeatures is not None:
            if matrix.shape[1] != self.nfeatures:
                raise AssertionError("additional and current objects has different numbers of features")
            self.objects = np.vstack([self.objects, matrix])
            self.nobjects += matrix.shape[0]
        else: # first add
            self.objects = matrix 
            self.nobjects, self.nfeatures = self.objects.shape

    def vif(self):
        if self.objects is None:
            raise ValueError("Can`t use this method for empty data")

        def _regression_residuals(x, y):
            """
            Calculate regression residuals:
            y - x * (x' * x)^(-1) * x' * y;
            Input:
                x - array(l, n)
                y - array(l, 1)
            Output:
                residuals y - x*w - array(l, 1)
            """
            return np.asarray(y - x * (x.T * x)**(-1) * x.T * y)

        vif = np.empty([self.nfeatures, 1])
        for i in xrange(self.nfeatures):
            rows = range(i) + range(i + 1, self.nfeatures)
            vif[i] = sum(np.asarray(self.objects[:, i] - np.mean(self.objects[:, i])) ** 2) /\
            sum(_regression_residuals(self.objects[:, rows], self.objects[:, i]) ** 2)

        return vif

    def belsley(self):
        if self.objects is None:
            raise ValueError("Can`t use this method for empty data")

        from scipy.linalg import svd

        [U,S,V] = svd(self.objects);

        # divide each row of V.^2 by S(i).^2 - singular value
        Q = np.dot(np.diag((1 / S) ** 2), V ** 2)

        # Normalize Q: column total is 1
        Q = np.dot(Q, np.diag(1 / sum(Q)))

        conditionality_indexes = max(S) / S[:,np.newaxis]
        return(conditionality_indexes, Q)

    def plot(self, features=(0, 1)):
        """
        Plot objects in 2D plane (f1, f2). features = (f1, f2)
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullFormatter

        x, y = self.objects[:, features[0]].flatten().tolist()[0], \
            self.objects[:, features[1]].flatten().tolist()[0]
        nullfmt = NullFormatter() # no labels

        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width + 0.02

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]

        # start with a rectangular Figure
        plt.figure(1, figsize=(8, 8))

        axScatter = plt.axes(rect_scatter)
        axHistx = plt.axes(rect_histx)
        axHisty = plt.axes(rect_histy)

        # no labels
        axHistx.xaxis.set_major_formatter(nullfmt)
        axHisty.yaxis.set_major_formatter(nullfmt)

        # the scatter plot:
        axScatter.scatter(x, y)

        # now determine nice limits by hand:
        binwidth = 0.25
        xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
        lim = ( int(xymax/binwidth) + 1) * binwidth

        axScatter.set_xlim((min(x), max(x)))
        axScatter.set_ylim((min(y), max(y)))

        bins = np.arange(-lim, lim + binwidth, binwidth)
        axHistx.hist(x, bins=bins)
        axHisty.hist(y, bins=bins, orientation='horizontal')

        axHistx.set_xlim( axScatter.get_xlim() )
        axHisty.set_ylim( axScatter.get_ylim() )

        plt.show()
  

class DataSet(object):
    def __init__(self, objects=None, labels=None):        
        if (objects is None) ^ (labels is None):
            raise TypeError("Require zero or two arguments for input")

        self.objects = ObjectFeatureMatrix()
        self.labels = None

        if objects is not None: # so labels is not None too
            self.add(objects, labels)

    def add(self, objects, labels):
        objects, labels = np.asmatrix(objects), np.asmatrix(labels)

        if labels.shape[0] == 1:
            labels = labels.T

        if labels.shape[1] != 1:
            raise AssertionError("One of the label dimensions must be 1")

        if labels.shape[0] != objects.shape[0]:
            raise AssertionError("Number of objects must be equal number of labels")

        if self.labels is None:
            self.labels = labels
        else:
            self.labels = np.vstack([self.labels, labels])
        self.objects.add(objects)

    def get_number_objects(self): 
        return self.objects.nobjects
    nobjects = property(get_number_objects, doc="return number of objects")

    def get_number_features(self): 
        return self.objects.nfeatures
    nfeatures = property(get_number_features, doc="return number of features")

    def plot(self, features=(0, 1)):
        """
        Plot data in 'features' axis.
        FIXIT: add more than 8 colors, specified by COLORS
        """
        import matplotlib.pyplot as plt
        COLORS="rbgcmykw"

        labels = np.asarray(self.labels.flatten())[0]
        uniq_labels = list(set(labels))

        x, y = self.objects.objects[:, features[0]],\
            self.objects.objects[:, features[1]]

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        # the scatter plot:
        axScatter = plt.subplot(111)
        # axScatter.set_aspect(1.) # square image

        for index, uniq_label in enumerate(uniq_labels):
            plt.plot(x[labels == uniq_label], y[labels == uniq_label], 
                '%so' % COLORS[index])

        # define picture size
        xlim, ylim = plt.xlim(), plt.ylim()

        # create new axes on the right and on the top of the current axes
        # The first argument of the new_vertical(new_horizontal) method is
        # the height (width) of the axes to be created in inches.
        divider = make_axes_locatable(axScatter)
        axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=axScatter)
        axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=axScatter)

        # make some axis - labels invisible
        plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(),
            visible=False)

        # now determine nice limits by hand:
        binwidth = 0.5
        xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
        lim = ( int(xymax/binwidth) + 1) * binwidth

        bins = np.arange(-lim, lim + binwidth, binwidth)

        x_bar, y_bar, color = [], [], []
        for index, uniq_label in enumerate(uniq_labels):
            x_bar.append(x[labels == uniq_label])
            y_bar.append(y[labels == uniq_label])
            color.append(COLORS[index])

        axHistx.hist(x_bar, bins=bins, color=color)
        axHisty.hist(y_bar, bins=bins, color=color, orientation='horizontal')

        # plot according to size
        width = xlim[1] - xlim[0]
        height = ylim[1] - ylim[0]
        # add +- 5% to border
        xlim = (xlim[0] - width / 20, xlim[1] + width / 20)
        ylim = (ylim[0] - height / 20, ylim[1] + height / 20)

        axScatter.set_xlim(xlim)
        axScatter.set_ylim(ylim)

        # plt.draw()
        plt.show()


if __name__ == "__main__":
    import doctest
    doctest.testfile("%s.test" % __file__.split(".", 1)[0])
