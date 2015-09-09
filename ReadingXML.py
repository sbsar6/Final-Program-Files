import pandas as pd
from tkinter import *
from tkinter import ttk
import xml.etree.ElementTree as ET 
#import numpy as np
#import matplotlib.mlab as mlab
import glob
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from lxml import etree
from AmpersandFix import *
from xml.etree import ElementTree
import re
import time


##path = "C:\\Users\\Andrew\\Desktop\\python data mining"
##dataPath = path +'\\GameData'
##game_list = [x[0] for x in os.walk(dataPath)]
##
##for each_file in game_list:
##    file_name = os.listdir(each_file)
##    print(file_name)
 
    

#fix method used to remove & sign for some games
#fix('150215 LvD.xml')
#indent method to make printing out elements from the tree in a prettier way
def indent(elem, level=0):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

#setting up a dataframe
df = pd.DataFrame(columns = ['Clock',
                             'TimeLeft',
                             'PhaseID',
                             'PhaseTime',
                             'PhaseName',
                             #'PhaseResult',
                             'RucksNo',
                             'PosGainline',
                             'NegGainline',
                             'Linebreak',
                             'LineBreakPrevPhase',
                             'rdNear',
                             'rdRuck',
                             'rdFar',
                             'rdMiddle',
                             'rsSlow',
                             'rsMedium',
                             'rsFast',
                             'KickTotal',
                             'KickedInfield',
                             'KickedtoTouch',
                             'turnoverTot',
                             'turnoverKick',
                             'turnoverContact',
                             'fromTurnover',
                             'penFK',
                             'penContact',
                             'Tries',
                             'TryTotal',
                             'PenTotal',
                             'TotDraPoss',
                             'TotOppPoss',
                             'TotDraLineB',
                             'TotAwayLineBreaks',
                             'DragScore',
                             'AwayScore'
                             
                             ])

                             

homePhases = [".:P1", ".:P2",".:P3",".:P4",".:P5",".:P6",".:P7",".:P8",".:P9",".:P10",":LINEOUT",
             ":PHASE BALL",":POSSESSION", ":RESTART RECEPTION", ":SCRUM", "Territory A","Territory B",
             "Territory C","Territory D", "OTHER"]
oppPhases = [".OP1", ".OP2",".OP3",".OP4",".OP5",".OP6",".OP7",".OP8",".OP9",".OP10","Opposition Lineout",
             "Opposition Other","Opposition Posession", "Opposition Restart Recept", "Opposition Scrum",
             "Opposition territory A","Opposition territory B","Opposition territory C","Opposition territory D",
             "OppositionPhase Ball"]
#Getting information from tree
for filename in glob.iglob("C:\\Users\\Andrew\\Desktop\\python data mining\\GameData/*.xml"):
    print(filename)
    tree = etree.parse(filename)
        
    
    
    Troot = tree.getroot()
    #print(etree.tostring("ID", pretty_print=True))
    #for element in Troot.iter("ID"):
    #    print("%s - %s" % (element.tag, element.text))


    preIdArr = []     
    #print (Troot[0][0].get("ID"))
    myArray = []
    tryStats=[]
    possStats= []
    count = 0
    gameClock = 0
    altClock= 0
    cStart = 0
    cEnd = 0
    halfTime = 0
    homeScore=0
    awayScore=0
    preEvent = ["None"]
    HomePossesTime =0
    OppPossesTime = 0
    TotHomeLinebreaks =0
    TotAwayLinebreaks =0
    ##for element in Troot.iter("ID"):
    ##    if element.text =='1':
    ##        print (etree.tostring(element, pretty_print=True))
                

    gameClock = 0
    cEnd=0
    cStart = 0
    highPhase = 0 
    
    for child in Troot:
        if child.tag != 'ALL_INSTANCES':
            continue
        #print (child.tag)
        #print (child.keys())
        
        for item in child:
                #i =0
                #print(item.tag)
                tryYN=0
                instanceText = item[3].text
                if any(match in instanceText for match in oppPhases):
                    HomeBall = False
                if any(match in instanceText for match in homePhases):
                    HomeBall = True


                    #works just not sure what to do with it
                    '''try:
                        OphaseNum = re.findall(r'\d+', instanceText)
                        OphaseNum2 = list(map(int, phaseNum))
                        if OphasNum2 > OhighPhase:
                            OhighPhase = phasNum2
                    except:
                        HighestPhase = 0'''
              
                if any(match in instanceText for match in homePhases):
                    HomeBall = True
                    #time.sleep(15)
                if instanceText =="Try":
                    tryYN=1
                    if HomeBall == True:
                        homeScore+=5
                        print("Home:", homeScore, "Away:", awayScore)
                    else:
                        awayScore +=5
                        print("Home:", homeScore, "Away:", awayScore)
                
                if instanceText =="! LINEBREAK":
                    
                    if HomeBall == True:
                        TotHomeLinebreaks +=1
                    else:
                        TotAwayLinebreaks +=1
                
                if instanceText =="HALF-TIME":
                    halfTime=  (float(item[1].text))/60

                posGain = 0
                negGain = 0
                linebreak = 0
                rdNear = 0
                rdRuck = 0
                rdFar = 0
                rdMiddle = 0
                numRucks = 0
                rsSlow =0
                rsMedium= 0
                rsFast= 0
                kicksTotal = 0
                kickInfield = 0
                kickTouch = 0
                turnoverTot = 0
                turnoverKick = 0
                turnoverContact=0
                fromTurnover=0
                penFK=0
                penContact=0
                prePhase = "N/A"
                tries=0
                
                penYN=0
                LineBreakTest= False
                for instance in item:


                    for event in instance:
                        #Tally up occurances
                        if event.text == ".CONVERSION":
                            if HomeBall == True:
                                homeScore+=2
                                print("Home:", homeScore, "Away:", awayScore)
                            else:
                                awayScore +=2
                                print("Home:", homeScore, "Away:", awayScore)
                        if event.text == ".PENALTY":
                            penYN=1
                            if HomeBall == True:
                                homeScore+=3
                                print("Home:", homeScore, "Away:", awayScore)
                            else:
                                awayScore +=3
                                print("Home:", homeScore, "Away:", awayScore)

                        if event.text == "Gainline +":
                            posGain +=1

                        if event.text == "Gainline -":
                            negGain +=1
                        if event.text =="Gain LineBreak":
                            #print('Event', event[-1].text)
                            LineBreakTest = True
                            linebreak+=1
                            preEvent.append(PreText)
                        if event.text =="RD - Near":
                            rdNear+=1
                        if event.text =="RD - RUCK":
                            rdRuck+=1
                        if event.text =="RD - Far":
                            rdFar+=1
                        if event.text =="RD - Middle":
                            rdMiddle+=1
                        if event.text =="RS-Average Ball":
                            rsMedium+=1
                        if event.text =="RS- Quick Ball":
                            rsFast+=1
                        if event.text =="RS-Slow Ball":
                            rsSlow+=1
                        if event.text =="Kicks Total":
                            kicksTotal+=1
                        if event.text =="Kicked Infield":
                            kickInfield+=1
                        if event.text =="Kicks to Touch":
                            kickTouch +=1
                        if event.text =="Turnover Total":
                            turnoverTot +=1
                        if event.text =="Turnover Kick":
                            turnoverKick +=1
                        if event.text =="Turnover Contact":
                            turnoverContact +=1
                        if event.text =="From Turnover":
                            fromTurnover +=1
                        if event.text =="Pen/FK":
                            penFK +=1
                        if event.text =="Pen - Contact Area":
                            penContact +=1
                        if event.text =="! Try":
                            tries +=1
                        #List the last event that led to linebreak    
                        PreText = event.text
                        if LineBreakTest == True:
                            prePhase = preEvent[-1]
                            #print('PrePhase', prePhase)
                            LineBreakTest = False


                    numRucks = rdNear+rdFar+rdMiddle+rdRuck      
                    nPhase= item[3].text
                    highPhase = 0    
                    #if re.match(r'.:P10+', nPhase):
                     #   print (etree.tostring(item, pretty_print=True))
                                       
                    #phaseNum = list(map(int, phaseNum))
                    starT = (float(item[1].text))/60
                    
                    endT = float(item[2].text)/60
                    phaseTime =  endT - starT
                    if endT > cEnd:

                        cTemp=0
                        if item[3].text == "HALF-TIME":
                            cTemp=0
                        elif starT>cEnd:
                                                         
                            cTemp = endT - starT
                            
                        else:
                            cTemp= endT - cEnd
                            #print('start not higher cTemp', cTemp)
                        cEnd= endT
                        
                        cStart =starT
                        
                        gameClock = gameClock + cTemp
                        #print ('Game Clock =',gameClock)
                  
                        
                    '''
                    if currTime > 20:
                        if currTime <21:
                            print('20 Mins Gone')
                    if currTime > 40:
                        if currTime <41:
                            print('40 Mins Gone')
                    if currTime > 60:
                        if currTime <61:
                            print('60 Mins Gone')
                    if currTime > 80:
                        if currTime <81:
                            print('80 Mins Gone')
                '''
                    
                      
                 
                     
                    if instance.text == ":POSSESSION":
                        pEnd = float(item[2].text)/60
                        
                        pStart = float(item[1].text)/60
                        
                        possT =  pEnd - pStart 
                        HomePossesTime +=possT
                    if instance.text == "Opposition Possession":
                        opEnd = float(item[2].text)/60
                        
                        opStart = float(item[1].text)/60
                        
                        opPossT =  pEnd - pStart 
                        OppPossesTime += opPossT
                        

                                           

                timeLeft = 0        
                df = df.append({'Clock':gameClock,
                                    'PhaseID':item[0].text,
                                    'PhaseTime':phaseTime,
                                    'PhaseName':nPhase,
                                    'TimeLeft':timeLeft,
                                    #'PhaseResult',
                                    'RucksNo':numRucks,
                                    'PosGainline':posGain,
                                    'NegGainline':negGain,
                                    'Linebreak':linebreak,
                                    'LineBreakPrevPhase':prePhase,
                                    'rdNear': rdNear,
                                    'rdRuck':rdRuck,
                                    'rdFar':rdFar,
                                    'rdMiddle':rdMiddle,
                                    'rsSlow':rsSlow,
                                    'rsMedium':rsMedium,
                                    'rsFast':rsFast,
                                    'KickTotal':kicksTotal,
                                    'KickedInfield':kickInfield,
                                    'KickedtoTouch':kickTouch,
                                    'turnoverTot':turnoverTot,
                                    'turnoverKick':turnoverKick,
                                    'turnoverContact':turnoverContact,
                                    'fromTurnover':fromTurnover,
                                    'Tries':tries,
                                    'penFK': penFK,
                                    'penContact': penContact,
                                    'TryTotal':tryYN,
                                    'PenTotal':penYN,
                                    'TotDraPoss':HomePossesTime,
                                    'TotOppPoss':OppPossesTime,
                                    'TotDraLineB':TotHomeLinebreaks,
                                    'TotAwayLineBreaks':TotAwayLinebreaks,
                                    'DragScore':homeScore,
                                    'AwayScore':awayScore
                                    #'Turnover',])
                                    }, ignore_index=True)

    fullTime = 0
    fullTime= cEnd
    for i, row in df.iterrows():
               
        tempN=float(df.ix[i,"Clock"])
        
        timeLeft = fullTime - tempN
        if timeLeft < 70:
            if timeLeft< 60:
                if timeLeft < 50:
                    if timeLeft < 40:
                        if timeLeft < 30:
                            if timeLeft< 20:
                                if timeLeft < 10:
                                    if timeLeft < 5:
                                        timeText = 5
                                    else: timeText = 10
                                else: timeText = 20
                            else: timeText = 30
                        else: timeText = 40
                    else: timeText = 50
                else: timeText = 60
            else: timeText = 70
        else: timeText = 80              
        try:
            df.ix[i,"TimeLeft"] = timeText
        except:
            print('Doesn''t work')
    print("Final Home:", homeScore, "Away:", awayScore)
    print ("HalfTime:", halfTime)
    print ("FullTime", fullTime)
    
df.to_csv('DragAllGames.csv')
#data = pd.read_csv("Game_Stats.csv")
#Data_Play_Ball = data[data["PhaseName"] == "BALL IN PLAY"]
#print(Data_Play_Ball)                     
#Data_Play_Ball.to_csv("BiP_Stats.csv")   
#Data_Possesion = data[data["PhaseName"] == ":POSSESSION"]
#print(Data_Play_Ball)                     
#Data_Possesion.to_csv("Possesion_Stats.csv")  
    
                   

#print ('TryStats: ' ,tryStats)
#print ('Poss Stats' , possStats)
#print ('Game Clock =',gameClock)

