import random
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
from bokeh.palettes import brewer
palette = brewer["Blues"][3]
from bokeh.plotting import *
from bokeh.charts import Bar, show, output_file
from bokeh.plotting import figure, HBox, output_file, show, VBox
from collections import OrderedDict

path = "EPSevenSeasonsTeamNames.csv"

df = pd.read_csv(path)
columns1 = ['Tot_Own_Scrums','Penalties_Won']
byTeamName = df.groupby('Team_Name')
scrums_penalties = byTeamName.sum()[columns1]
print (scrums_penalties.corr())
#print (scrums_penalties)

TeamNames = df.Team_Name.unique()
#print(TeamNames)
output_file("markers.html")
#print(scrums_penalties["Team_Name"])
palette = list(reversed([
    "#f7f7f7","#d1e5f0","#228B22","#FFA500","#FFB6C1","#f4a582","#fddbc7","#f7f7f7","#d1e5f0","#1E90FF","#DC143C","#00FFFF","#808000","#9400D3","#ADFF2F","#FFFF00","#FF0000","#FF1493"
]))

palette2 = list(reversed([
   "#1E90FF","#6B8E23"]))
p = figure(title="Penalties Won and Own Scrums by Teams in EP", x_axis_label = "Total Own Scrums",
       y_axis_label = "Penalties Awarded")


p.circle(scrums_penalties['Tot_Own_Scrums'], scrums_penalties['Penalties_Won'], size=12,
       color=palette, line_color="black", fill_alpha=0.8)

'''p.text(scrums_penalties['Tot_Own_Scrums'], scrums_penalties['Penalties_Won']+0.3,
    text=TeamNames,text_color="#333333",
    text_align="center", text_font_size="10pt")'''

#mtext(p, [2.5], [0.5], "circle / o")
show(p)  # open a browser

#Average Penalties Awarded by Referee
columns2 = ['Penalties_Won']
byReferee = df.groupby('Referee')

penalties_sum = byReferee.sum()[columns2]
penalties_mean= byReferee.mean()[columns2]
#print(penalties_mean)
penalties_count = byReferee.count()[columns2]
bar2 = Bar(penalties_mean,title="Penalties Awarded by Referee",palette=brewer["Reds"][3], stacked=True)
    
output_file("Referee Penalties.html")
show(bar2)

#Penalties by Team 
bar = Bar(scrums_penalties,title="Penalties and Own Scrums Won by Team",palette=brewer["Purples"][3], stacked=True)
    
output_file("rankings.html")
show(bar)


PlayingLocationAway = df[df['Playing_Location'] == 0]
PenaltiesAway = PlayingLocationAway['Penalties_Won']

Away = [random.random()*20 for x in range(len(PenaltiesAway))]
#print(Away)

PlayingLocationHome = df[df['Playing_Location'] == 1]
PenaltiesHome = PlayingLocationHome['Penalties_Won']


#Average Penalties Awarded by Referee Home
columns3 = ['Penalties_Won']
columns4 = ['Penalties_Conceeded']
byRefereeHome = PlayingLocationHome.groupby('Referee')
penalties_sumHome = byRefereeHome.sum()[columns3]
#print (penalties_sumHome)
penalties_meanHome= byRefereeHome.mean()[columns3]
penalties_countHome = byRefereeHome.count()[columns3]

#Average Penalties Awarded by Referee Away
byRefereeAway = PlayingLocationAway.groupby('Referee')
penalties_sumAway = byRefereeAway.sum()[columns3]
#print (penalties_sumAway)
penalties_meanAway= byRefereeAway.mean()[columns3]
penalties_countAway = byRefereeAway.count()[columns3]

#Combining Data
New_DF = pd.DataFrame(columns=["Penalties Home","Penalties Away"])
New_DF["Penalties Home"] = penalties_meanHome["Penalties_Won"]
New_DF["Penalties Away"] = penalties_meanAway["Penalties_Won"]


values = OrderedDict( PenaltiesHome=New_DF["Penalties Home"].tolist(),
                      PenaltiesAway=penalties_meanAway["Penalties_Won"].tolist(), 
                    )

bar4 = Bar(values,New_DF.index.tolist(),palette=palette2,legend=True)
show(bar4)
