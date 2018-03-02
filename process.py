import os
import matplotlib.pyplot as plt
import scipy
import scipy.io.wavfile

from os import listdir
from os.path import isfile, join
from pylab import np
from scikits.talkbox.features import mfcc
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

## Reading Files
print('# Reading Files...')
samples_files = []
sample_folders = [f for f in listdir("samples") if not isfile(join(".", f))]
for path in sample_folders:
    samples_files.append([path+'/'+f for f in listdir('samples/'+path) if isfile(join('samples/'+path, f))])

## Labeling Samples
county = [len(f) for f in samples_files]
yf = range(5)
y=[]
for i in yf:
    for j in xrange(county[i]):
        y.append(i)

XA = []
ya = []

## Processing files
def create_fft(fn, myclass):
    try:
        sample_rate, X = scipy.io.wavfile.read(fn)
    except ValueError:
        return
    ceps, mspec, spec = mfcc(X)
    num_ceps = len(ceps)
    x = np.mean(ceps[int(num_ceps*1/10):int(num_ceps*9/10)], axis=0)
    y = np.int(myclass)
    XA.append(x)
    ya.append(y)

print('# Processing Frequency Coeficients...')
yiter = iter(y)
cwd = os.getcwd()
for myclassfiles in samples_files:
    for myfilepath in myclassfiles:
        create_fft(cwd+'/samples/'+myfilepath, yiter.next())

## Fitting algorithm
print('# Training NN...')
X_train, X_test, y_train, y_test = train_test_split(XA, ya, test_size=0.33, random_state=1)

## Prediction
clf = LogisticRegression()
print('# Using NN for prediction...')
clf.fit(X_train, y_train)
y_pred = clf.fit(X_train, y_train).predict(X_test)

# Confusion Matrix
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(sample_folders))
    plt.xticks(tick_marks, sample_folders, rotation=45)
    plt.yticks(tick_marks, sample_folders)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Processing results
print('# Plotting results...')
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)

# Cross Validation
scores = cross_validation.cross_val_score(clf, XA, ya, cv=3, scoring='accuracy')
print(scores[0])
