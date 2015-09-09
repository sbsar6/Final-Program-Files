
import random
import time
import re
import pandas as pd


import requests
import re
from bs4 import BeautifulSoup


home =""
away=""
urlList = []


soup = None

def https():
       
        for i in range (201245,201492,2):
            temp = ("http://en.espn.co.uk/super-rugby-2014/rugby/match/" + str(i)+".html?view=scorecard")
            urlList.append(temp)
        for i in range (170144,170269,1):
            temp = ("http://en.espn.co.uk/super-rugby-2013/rugby/match/" + str(i)+".html?view=scorecard")
            urlList.append(temp)
        for i in range (151446,151394,1):
            temp = ("http://en.espn.co.uk/super-rugby-2012/rugby/match/" + str(i)+".html?view=scorecard")
            urlList.append(temp)

https()

Data = pd.DataFrame(columns = ['Team_Name',
                                   'Tries Scored',
                                   'Conversions',
                                   'Conversion Attempts',
                                   'Conversion Percentage',
                                   'Penalty Goals',
                                   'Penalty Attempts',
                                   'Penalty Percentage',
                                   'Kick at Goal Success',
                                   'Drop Goals',
                                   'Drop Goal Attempts',
                                   'Kicks from Hand',
                                   'Passes',
                                   'Runs',
                                   'Metres Run',	
                                   'Possession(%)',
                                   'Clean Breaks',
                                   'Defenders Beaten',
                                   'Offloads',
                                   'Rucks Won',
                                   'Total Rucks',
                                   '%Rucks Won',
                                   'Mauls Won',
                                   'Total Mauls',
                                   '%Mauls Won',
                                   'Turnovers Conceded',
                                   'Tackles Made',
                                   'Tackles Missed',
                                   'Tackling Success Rate',
                                   'Own Scrum Won',
                                   'Own Scrum Lost',
                                   'Opp Scrum Won',
                                   'Opp Scrum Lost',
                                   '%Scrums on own feed',
                                   'Own Lineout Won', 
                                   'Own Lineout Lost',
                                   'Opp Lineout Won',
                                   'Opp Lineout Lost',
                                   '%Lineouts on own feed',
                                   'Penalties Conceeded',
                                   'Yellow Cards',
                                   'Red Cards',
                                   'Referee',
                                   'Playing Location',
                                   'Result',
                                   'Outcome'
                                   ])
                                   
#urlList = ["http://en.espn.co.uk/super-rugby-2014/rugby/match/201481.html?view=scorecard","http://en.espn.co.uk/super-rugby-2014/rugby/match/201483.html?view=scorecard" ]                                   
for item in urlList: 
    #req = urllib2.Request(item, headers=hdr)
    #page = urllib2.urlopen(req)

    r = requests.get(item)
    page = r.text
    soup = BeautifulSoup(page)
    



    try: matchDet = soup.find(text ="Match stats").findNext().text
    except: continue

    regex = '</span> (.+?) <span class="liveSubNavText2">'
    pattern = re.compile(regex)
    try:
        result = re.findall(pattern,str(soup))[0]
        score = result.strip().split('-')
    except:
        homeScr= ((int(List[3][:1])*5) + (int(List[6][:1])*2)+(int(List[9][:1])*3)+(int(List[15][:1])*3))
     
        awayScr= ((int(List[5][:1])*5) + (int(List[8][:1])*2)+(int(List[11][:1])*3)+(int(List[17][:1])*3))
        #print('Score: ',homeScr, awayScr)
        score= [homeScr, awayScr]
        #print(score)
    

    #PreMatchBits = matchDet.find(text = "Pos").findAllPrevious('tr')
    FList = []
    List = []
    
    List = matchDet.replace('\t','').split('\n')
    #print(List)
    #replace('\n\t','')
    setPieceList =[]
    for item in List:
            if item == '  0 won, 0 lost':
                        setPieceList.append([0,'setPiece',0])
                        item = '(0%)'
                      
            if item.find("lost") != -1:
                    #print(item)
                    try:won = int(item[2:4])
                    except:won = int(item[2:3])
                    #print('won',won)
                    try:lost = int(item[10:12])
                    except:lost = int(item[9:11])
                    #print(lost)
                    setPieceList.append([won,'setPiece',lost])
                    List.remove(item)
      
    while '' in List:
            List.remove('')
    tmp = []
        
    regex = '</span> (.+?) <span class="liveSubNavText2">'
    pattern = re.compile(regex)
    try:
        result = re.findall(pattern,str(soup))[0]
        score = result.strip().split('-')
    except:
        homeScr= ((int(List[3][:1])*5) + (int(List[6][:1])*2)+(int(List[9][:1])*3)+(int(List[15][:1])*3))
     
        awayScr= ((int(List[5][:1])*5) + (int(List[8][:1])*2)+(int(List[11][:1])*3)+(int(List[17][:1])*3))
        
        score= [homeScr, awayScr]
      
    #print(len(List))
    List[1] = 'Teams'
    
    if len(List)== 83:
            ownRucksWon= int(List[47][:2])
            totOwnRucks = int(List[47][8:10])
            oppRucksWon= int(List[49][:2])
            totOppRucks = int(List[49][8:10])
            ownMaulsWon= int(List[50][:2])
            totOwnMauls = int(List[50][7:9])
            oppMaulsWon= int(List[53][:2])
            totOppMauls = int(List[53][7:9])

    elif len(List)== 80: 
            ownRucksWon= int(List[44][:2])
            totOwnRucks = int(List[44][8:10])
            oppRucksWon= int(List[46][:2])
            totOppRucks = int(List[46][8:10])
            ownMaulsWon= int(List[47][:2])
            totOwnMauls = int(List[47][7:9])
            oppMaulsWon= int(List[50][:2])
            totOppMauls = int(List[50][7:9])

    elif len(List)== 77: 
            ownRucksWon= int(List[41][:2])
            totOwnRucks = int(List[41][8:10])
            oppRucksWon= int(List[43][:2])
            totOppRucks = int(List[43][8:10])
            ownMaulsWon= int(List[44][:2])
            totOwnMauls = int(List[44][7:9])
            oppMaulsWon= int(List[47][:2])
            totOppMauls = int(List[47][7:9])
    maulList = []
    maulList.append([ownRucksWon, 'Rucks Won',oppRucksWon])
    maulList.append([totOwnRucks, 'Tot Rucks',totOppRucks])
    maulList.append([ownMaulsWon, 'Mauls Won',oppMaulsWon])
    maulList.append([totOwnMauls, 'Tot Mauls ',totOppMauls])  
    #print(maulList)
    for i in List:
                if i == '':
                     continue
                if i == 'Kick/pass/run':
                     
                     
                     continue
                elif i =='Attacking':
                     continue
                elif i == 'Defensive':
                     continue
                elif i =='Set pieces':
                    continue
                elif i == 'Discipline':
                    continue
                #elif i == 'Conversions':
                    
                #elif i == 'Mauls won':
                    #print (tmp[1])
                    #tmp[1]= 'Mauls won'
                   
               #     continue
               # elif i == 'Turnovers conceded':
                    #print(FList)
                    #tmp[0] = str(tmp[1])
                    #tmp[1] = 'Turnovers conceded'
               #     continue
                
                
                try:
                    try:
                        value = int(i.strip().replace('\n\t',''))
                    except Exception as e:
                        value = i.strip().replace('\n\t','')
                    if not isinstance(value,int):
                        tmp.append(float(i.strip().replace('\n\t','').split('%')[0].split("(")[1])/100.0)
                    else:
                        tmp.append(value)
                except Exception as e:
                    try:
                        tmp.append(float(i.strip().replace('\n\t','').split('%')[0])/100.0)
                    except Exception as e:
                        tmp.append(i.strip().replace('\n\t',''))
                while '' in tmp:
                    tmp.remove('')
                
                if len(tmp) == 3:
                    FList.append(tmp)
                    tmp = []
    
    
    CL = []
    for item in FList:
        if item[1] == 'Tackles made/missed':
            CL.append([item[0].split('/')[0],'Tackles made',item[2].split('/')[0]])
            CL.append([item[0].split('/')[1],'Tackles missed',item[2].split('/')[1]])
        elif item[1] == 'Yellow/red cards':
            
            CL.append([item[0].split('/')[0],'Yellow cards',item[2].split('/')[0]])
            CL.append([item[0].split('/')[1],'Red cards',item[2].split('/')[1]])
        elif item[1] == 'Territory (1H/2H)':
            pass
        else:
            CL.append(item)
    try:
        referee = soup.find(text ="Referee").findNext('a').text
    except: referee = 'Unknown'
    refList = [referee, "Referee", referee]
    CL.append(refList)    

    
    CL.append(["home","Location", "away"])
    lists = CL
    lists.append([score[0],"result",score[1]])
    if (int(score[0]) > int(score[1])):
        lists.append([1,"win",0])
    elif (int(score [0]) < int(score[1])):
        lists.append([0,"win",1])
    else: lists.append([3, 'win', 3])    
    


    scrumList=[]
    scrumList.append([setPieceList[0][0], "Own Won", setPieceList[1][0]])
    scrumList.append([setPieceList[0][2], "Own Lost", setPieceList[1][2]])
    scrumList.append([setPieceList[1][2], "Opp Won", setPieceList[0][2]])
    scrumList.append([setPieceList[1][0], "Opp Lost", setPieceList[0][0]])
    scrumList.append([setPieceList[2][0], "Own L Won", setPieceList[3][0]])
    scrumList.append([setPieceList[2][2], "Own L Lost", setPieceList[3][2]])
    scrumList.append([setPieceList[3][2], "Opp L Won", setPieceList[2][2]])
    scrumList.append([setPieceList[3][0], "Opp L Lost", setPieceList[2][0]])
    posession = 0
    cleanBreaks = 0
    defBeat = 0
    offloads = 0
    percentRucksWon = 0
    percentMaulsWon = 0
    turnovCon = 0
    tackMade = 0
    tackMissed =0
    tackSuccess = 0
    ownScrums =0
    ownLineouts = 0
    penConceed = 0
    yellCards =0
    redCards = 0
    ref = ""
    location = ""
    result =0
            
    
    #print(lists) 
    for i in range (0,3,2):
        
        conversions = int(lists[2][i][:2])
        
        conversionAttemps = int(lists[2][i][-2:])
        
        if conversionAttemps ==0:
            percentConversions = 0
        else: percentConversions = (conversions/conversionAttemps)*100
       
        penConversions = int(lists[3][i][:2])
        penConversionAttemps = int(lists[3][i][-2:])
        if penConversionAttemps ==0:
            penPercentConversions = 0
        else: penPercentConversions = (penConversions/penConversionAttemps)*100
        try:dropGoals = int(lists[5][i][:1])
        except:dropGoals = lists[5][i]
        try: dropGoalAttemps = int(lists[5][i][3:4])    
        except: dropGoalAttemps = 0
        kickSuccess = lists[4][i]
        kicksHand = lists[6][i]
        passes = lists[7][i]         
        runs = lists[8][i]
        metrun = lists[9][i]
        ownScrumWon = scrumList[0][i]
        ownScrumLost = scrumList[1][i]
        oppScrumWon = scrumList[2][i]
        oppScrumLost = scrumList[3][i]
        ownLineWon = scrumList[4][i]
        ownLineLost = scrumList[5][i]
        oppLineWon = scrumList[6][i]
        oppLineLost = scrumList[7][i]
        rucksWon = maulList[0][i]
        totRucks = maulList[1][i]
        maulsWon = maulList[2][i]
        totMauls = maulList[3][i]
        try:
            Data= Data.append({'Team_Name':lists[0][i],
                           'Tries Scored':lists[1][i],
                           'Conversions':conversions,
                           'Conversion Attempts': conversionAttemps,
                           'Conversion Percentage': percentConversions,
                           'Penalty Goals':penConversions,
                           'Penalty Attempts':penConversionAttemps,
                           'Penalty Percentage':penPercentConversions,
                           'Kick at Goal Success':kickSuccess,
                           'Drop Goals':dropGoals,
                           'Drop Goal Attempts':dropGoalAttemps,
                           'Kicks from Hand':kicksHand,
                           'Passes':passes,
                           'Runs': runs,
                           'Metres Run':metrun,	
                           'Possession(%)':lists[10][i],
                           'Clean Breaks':lists[11][i],
                           'Defenders Beaten':lists[12][i],
                           'Offloads':lists[13][i],
                           'Rucks Won': rucksWon,
                           'Total Rucks': totRucks,
                           '%Rucks Won':lists[14][i],
                           'Mauls Won': maulsWon,
                           'Total Mauls': totMauls,
                           '%Mauls Won':lists[15][i],
                           'Turnovers Conceded':lists[16][i],
                           'Tackles Made':lists[17][i],
                           'Tackles Missed':lists[18][i],
                           'Tackling Success Rate':lists[19][i],
                           'Own Scrum Won':ownScrumWon,
                           'Own Scrum Lost':ownScrumLost,
                           'Opp Scrum Won':oppScrumWon,
                           'Opp Scrum Lost':oppScrumLost,
                           '%Scrums on own feed':lists[20][i],
                           'Own Lineout Won':ownLineWon, 
                           'Own Lineout Lost':ownLineLost,
                           'Opp Lineout Won':oppLineWon,
                           'Opp Lineout Lost':oppLineLost,
                           '%Lineouts on own feed':lists[21][i],
                           'Penalties Conceeded':lists[22][i],
                           'Yellow Cards':lists[23][i],
                           'Red Cards':lists[24][i],
                           'Referee':lists[25][i],
                           'Playing Location':lists[26][i],
                           'Result':lists[27][i],
                           'Outcome':lists[28][i]
                       }, ignore_index = True)
            
            
        except:
            Data= Data.append({'Team_Name':lists[0][i],
                           'Tries Scored':lists[1][i],
                           'Conversions':conversions,
                           'Conversion Attempts': conversionAttemps,
                           'Conversion Percentage': percentConversions,
                           'Penalty Goals':penConversions,
                           'Penalty Attempts':penConversionAttemps,
                           'Penalty Percentage':penPercentConversions,
                           'Kick at Goal Success':kickSuccess,
                           'Drop Goals':dropGoals,
                           'Drop Goal Attempts':dropGoalAttemps,
                           'Kicks from Hand':kicksHand,
                           'Passes':passes,
                           'Runs': runs,
                           'Metres Run':metrun,	
                           'Possession(%)':0,
                           'Clean Breaks':lists[10][i],
                           'Defenders Beaten':lists[11][i],
                           'Offloads':lists[12][i],
                           'Rucks Won': rucksWon,
                           'Total Rucks': totRucks,
                           '%Rucks Won':lists[13][i],
                           'Mauls Won': maulsWon,
                           'Total Mauls': totMauls,
                           '%Mauls Won':lists[14][i],
                           'Turnovers Conceded':lists[15][i],
                           'Tackles Made':lists[16][i],
                           'Tackles Missed':lists[17][i],
                           'Tackling Success Rate':lists[18][i],
                           'Own Scrum Won':ownScrumWon,
                           'Own Scrum Lost':ownScrumLost,
                           'Opp Scrum Won':oppScrumWon,
                           'Opp Scrum Lost':oppScrumLost,
                           '%Scrums on own feed':lists[19][i],
                           'Own Lineout Won':ownLineWon, 
                           'Own Lineout Lost':ownLineLost,
                           'Opp Lineout Won':oppLineWon,
                           'Opp Lineout Lost':oppLineLost,
                           '%Lineouts on own feed':lists[20][i],
                           'Penalties Conceeded':lists[21][i],
                           'Yellow Cards':lists[22][i],
                           'Red Cards':lists[23][i],
                           'Referee':lists[24][i],
                           'Playing Location':lists[25][i],
                           'Result':lists[26][i],
                           'Outcome':lists[27][i]
                       }, ignore_index = True)
            

    #print(dictionary)
    print (lists[0][0], score[0], ',',lists[0][2], score[1])
    
    #Data.append(dictionary, ignore_index = True)
teamDict ={} 
i=0 
for item in Data.Team_Name.unique(): 
    teamDict[item] = i 
    i+=1 
print(teamDict)
for item in Data.Team_Name.unique():

    tempI = str(item)
    
    tempN = str(teamDict[item])
    
    Data['Team_Name'] = Data['Team_Name'].apply(lambda x: str(x).replace(tempI,tempN))
Data.columns = [c.replace(' ', '_') for c in Data.columns]
Data['Referee'] = 0    
Data['Tries_Scored'] = Data['Tries_Scored'].apply(lambda x:float(str(x).split("(")[0]))
Data['Tot_Own_Scrums'] = Data['Own_Scrum_Won']+ Data['Own_Scrum_Lost']
Data['Playing_Location'] = Data['Playing_Location'].apply(lambda x: str(x).replace("home","1")[0])
Data['Playing_Location'] = Data['Playing_Location'].apply(lambda x: str(x).replace("a","0")[0])
Data['%Scrums_on_own_feed'] = Data['%Mauls_Won'].apply(lambda x: str(x).replace("0 won, 0 lost","0")[0])

Data['Penalties_Conceeded'] = Data['Penalties_Conceeded'].apply(lambda x: (str(x)[:2]))
C_List = Data["Penalties_Conceeded"].tolist()
#print(C_List)
for i in range(0,len(C_List),2):
    try:
        tmp =  C_List[i]
        C_List[i] = C_List[i+1]
        C_List[i+1]= tmp
        
        #print("New List:", C_List)
    except Exception as e:
         pass

Data["Penalties_Won"] = C_List

Data.to_csv("SR3Season.csv")



