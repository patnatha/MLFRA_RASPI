import requests
import json
import sqlite3
import time
import threading
import signal
import sys

tokenFile = "/home/pi/Documents/MLFRA/token.auth"
postUrl = "https://redcap.wakehealth.edu/redcap/api/"
dbFile = "backup.db"
tableName = "to_log"

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
            return(str(int(theVal)))
        except Exception as err:
            print(err)
            return(None)

def convert_two_decimal(theVal):
    if(theVal == None):
        return(None)
    else:
        try:
            return(str("{:.2f}".format(theVal)))
        except:
            return(None)


def create_sqlite_table():
    conn = sqlite3.connect(dbFile)
    cur = conn.cursor()
    sql = "CREATE TABLE IF NOT EXISTS " + tableName + " (record_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, the_text TEXT);"
    cur.execute(sql)
    conn.commit()
    conn.close()
    
def log_later(theDatas):
    create_sqlite_table()

    conn = sqlite3.connect(dbFile)
    cur = conn.cursor()

    sql = 'INSERT INTO ' + tableName + ' (the_text) VALUES("' + json.dumps(theDatas).replace('"', '\'') + '");'
    #print(sql)
    cur.execute(sql)

    conn.commit()
    conn.close()

stop_event= threading.Event()
def survail_db_to_upload():
    create_sqlite_table()

    while True:
        conn = sqlite3.connect(dbFile)
        cur = conn.cursor()

        sql = "SELECT record_id, the_text FROM " + tableName
        toDel = []
        for row in cur.execute(sql):
            theDatas = json.loads(row[1].replace("'","\""))
            #print(theDatas)
            postRes = post_redcap(theDatas)
            if(postRes == 1):
                toDel.append(str(row[0]))
        conn.commit()
        conn.close()

        print("SQLITE Posted:", len(toDel))
        conn = sqlite3.connect(dbFile)
        cur = conn.cursor()
        for itemDel in toDel:
            sql = "DELETE FROM " + tableName + " WHERE record_id = " + itemDel
            cur.execute(sql)
        conn.commit()
        conn.close()

        time.sleep(60)

        #Kill thread
        if(stop_event.is_set()):
            sys.exit(0)

survailThread = threading.Thread(target=survail_db_to_upload, args=())
survailThread.daemon = False
survailThread.start()

def signal_term_handler(signal, frame):
    stop_event.set()
    print("Killing Thread")
    sys.exit(0)
catchable_sigs = set(signal.Signals) - {signal.SIGKILL, signal.SIGSTOP}
for sig in catchable_sigs:
    signal.signal(sig, signal_term_handler) 

def post_redcap(theDatas):
    try:
        data = {
            'token': loaded_token,
            'content': 'record',
            'action': 'import',
            'format': 'json',
            'type': 'flat',
            'overwriteBehavior': 'normal',
            'forceAutoNumber': 'true',
            'data': '',
            'returnContent': 'count',
            'returnFormat': 'json'
        }

        theSendStruct = {'record_id': 0}
        for key in theDatas:
            theSendStruct[key] = theDatas[key]
        data['data'] = json.dumps([theSendStruct])

        #log_later(theDatas)
        #return
        r = requests.post(postUrl, data=data)
        
        if(r.status_code != 200):
            print(r.json())
            log_later(theDatas)
            return(-2)
        elif(r.status_code == 200 and r.json()['count'] != 1):
            print(r.json())
            log_later(theDatas)
            return(-1)
        else:
            return(1)
    except Exception as err:
        print("post_redcap:", err)
        log_later(theDatas)
        return(-3)

#post_redcap({'name': '2021-10-28_3', 
#             'datetime': '2021-10-28 10:09:30',
#             'yn': 1, 'probability': 86.25, 'elapsed_time': 3.45})

