import numpy
import pandas
from matplotlib import pyplot
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks

init_notebook_mode(connected=True)
cufflinks.go_offline()
plotly.tools.set_credentials_file(username='tamoghnabhattacharjee', api_key='qS2Jvxst15TGZXmqsujM')

df = pandas.DataFrame(numpy.random.randn(100, 4), columns='A B C D'.split())

# df.iplot(kind='scatter', x='A', y='B')

