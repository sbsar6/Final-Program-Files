import random
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
style.use("ggplot")

path = "Top14OneSeason.csv"

data = pd.read_csv(path)
data.head()

PlayingLocationAway = data[data['Playing_Location'] == 0]
PenaltiesAway = PlayingLocationAway['Penalty_Attempts']
print(PenaltiesAway)
Away = [random.random()*20 for x in range(len(PenaltiesAway))]
print(Away)
fig = plt.figure()
PlayingLocationHome = data[data['Playing_Location'] == 1]
PenaltiesHome = PlayingLocationHome['Penalty_Attempts']
Home = [(50 + random.random()*20) for x in range(len(PenaltiesAway))]
ax = fig.add_subplot(111)
Away = plt.scatter(Away,PenaltiesAway, color="purple")
Home = plt.scatter(Home,PenaltiesHome, color="blue")
plt.legend((Away,Home),("Away, ","Home"),loc="upper center")
plt.title("Top 14: Penalty Attempts by Playing Location")
totHome = str("Tot. Home Pens. "+ str(sum(PenaltiesHome)))
totAway = str("Tot. Away Pens "+str(sum(PenaltiesAway)))
ax.text(0.01, 0.01, totAway,
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax.transAxes, fontsize=15)
ax.text(0.95, 0.01, totHome,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes, fontsize=15)
plt.show()
