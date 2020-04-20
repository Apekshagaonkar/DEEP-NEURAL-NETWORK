try:
    from tabula import read_pdf,convert_into
except:
    print("install tabula for converting from pdf to csv")
import pandas as pd
import os
import shutil

#try:
    #convert_into("Assignment_2/ACUTE_INFARCTS_Updated.pdf","Assignment_2/ACUTE_INFARCTS_Updated.csv",output_format="csv",pages='all')
#except:
    #print("FILE ALREADY EXISTS")
df =pd.read_csv('Assignment_2/ACUTE_INFARCTS_Updated.csv')

try:
    os.mkdir('CLEANED_DATA')
except:
    print("cleaned_data file is already created")

x=df[['Sl.','Location of acute infarct']]
y=x.to_numpy()
print(y)

x=os.getcwd() #should be in final directory
print(x)

j=0
for i in y:
  if i[1] not in os.listdir('CLEANED_DATA'):
    print(os.listdir('CLEANED_DATA'))
    os.chdir('CLEANED_DATA')
    try:
        os.mkdir(i[1])
    except:
        print("file already exits ", i[1])
    os.chdir('../')
  os.chdir('CLEANED_DATA/'+str(i[1]))
  dest=os.getcwd()
  #SOURCE = path of dataset provided that needs to be categorised based on location
  os.chdir('../')
  os.chdir('../')
  os.chdir('Assignment_2')
  source="Case "+str(int(i[0]))
  for i in os.listdir(source):
      os.chdir(source)
      #w=str(j)+'.jpg'
      #os.rename(i,w)
      #i=w
      newPath = shutil.copy(i,dest)
      os.chdir('../')
      j=j+1
  os.chdir(x)
os.getcwd()
# set the path to the final folder
os.chdir('../')
os.chdir('../')