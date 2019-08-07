import pandas
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

df = pandas.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
x = df.iloc[:, 0].values
y = df.iloc[:, 1].values

# cleaning text

nltk.download('stopwords')
ps = PorterStemmer()
for i in range(x.shape[0]):
    words = re.sub('[^a-zA-Z]', ' ', x[i]).split()
    x[i] = [ps.stem(wrd) for wrd in words if wrd not in set(stopwords.words('english'))]
    x[i] = ' '.join(x[i])


cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(x).toarray()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)

classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
