import requests
import json

tokenFile = "/home/pi/Documents/MLFRA/token.auth"
postUrl = "https://redcap.wakehealth.edu/redcap/api/"

def get_token():
    f = open(tokenFile)
    theToken = f.read().strip("\n")
    f.close()
    return(theToken)
loaded_token = get_token()

def convert_int(theVal):
    if(theVal == None):
        return(None)
    else:
        try:
            return(str(int(round(theVal, 0))))
        except:
            return(None)

def convert_two_decimal(theVal):
    if(theVal == None):
        return(None)
    else:
        try:
            return(str(round(theVal,2)))
        except:
            return(None)


def post_redcap(theDatas):
    data = {
        'token': loaded_token,
        'content': 'record',
        'action': 'import',
        'format': 'json',
        'type': 'flat',
        'overwriteBehavior': 'normal',
        'forceAutoNumber': 'true',
        'data': '',
        'returnConcent': 'count',
        'returnFormat': 'json'
    }

    theSendStruct = {'record_id': 0}
    for key in theDatas:
        theSendStruct[key] = theDatas[key]
    data['data'] = json.dumps([theSendStruct])


    r = requests.post(postUrl, data=data)
    
    if(r.status_code != 200):
        print(r.json())
        return(-2)
    elif(r.status_code == 200 and r.json()['count'] != 1):
        print(r.json())
        return(-1)
    else:
        return(1)

post_redcap({'name': '2021-10-28_3', 
             'datetime': '2021-10-28 10:09:30',
             'yn': 1, 'probability': 86.25, 'elapsed_time': 3.45})

