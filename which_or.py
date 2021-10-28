from datetime import datetime

def returnName(theDatetime):
    theName = theDatetime.strftime("%Y-%m-%d")
    theName += "_3" #this is the OR for this logger
    return(theName)

