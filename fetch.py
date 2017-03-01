import requests
import urllib,json
import pandas as pd
import string
final_dict={}
id_list=[]
url_list = []
summary_list=[]
title_list=[]
answer_list = []
def getPageCount():
    response = requests.get("https://ask.openstack.org/en/api/v1/users/")
    data = response.json()
    return int(data['pages'])

def fetchData(pages):
 for page in range(1, 10):
    parameters = {"scope": all, "page": page}
    response = requests.get("https://ask.openstack.org/en/api/v1/questions/", params = parameters)
    data = response.json()
    answer = 0
    for i in range(0,len(data['questions'])):
       for key,value in data['questions'][i].iteritems():
          url = data['questions'][i]['url'].encode('utf-8').strip()
          if 'url' in key:
             value = value.encode('utf-8').strip()
             url_list.append(value)
          elif 'summary' in key:
             value = value.encode('utf-8').strip()
             summary_list.append(value)
          elif 'title' in key:
             value = value.encode('utf-8').strip()
             title_list.append(value)
          elif 'accepted_answer_id' in key:
              if value is None:
                 answer = "-"
              else: answer=getAnswer(value,url)
              answer_list.append(answer)
              
    print "Completed Page#: "+str(page)
 final_dict['url'] = url_list
 final_dict['title'] = title_list
 final_dict['summary'] = summary_list
 final_dict['answer'] = answer_list
 buildCsv(final_dict)  

def getAnswer(answer_id,url):
   try:
      answer_id = str(answer_id)
      url_text = url+"?"+"answer="+answer_id+"#post-id-"+answer_id
      response = requests.get(url_text)
      import bs4
      soup = bs4.BeautifulSoup(response.text)
      id = "js-post-body-"+answer_id
      answer = soup.select('div#'+id+' p')
      return answer
   except:
      print url

   

def buildCsv(final_dict):
 data_frame = pd.DataFrame(final_dict)
 data_frame.to_csv("ask_os.csv",sep=",")

def main():
     pages = getPageCount()
     fetchData(pages)

if __name__ == "__main__": main()
