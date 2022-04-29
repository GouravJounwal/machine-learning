#Lab 2: Demonstrate application of Dictionary by taking birthday month as an input and count number of birthdays in particular month by taking month as key . identify max number and minimum number of birthday

birthday={"jan":0,"feb":0,"mar":0,"apr":0,"may":1,"june":0,"july":0,"aug":0,"sep":0,"oct":0,"nov":0,"dec":0}
while(True):
  mon=input()
  if mon =="":
    break
  else:
    birthday[mon]=birthday[mon]+1

minv=birthday["jan"]
for i in birthday.keys():
  if birthday[i]<minv:
    minv=birthday[i]

maxv=birthday["jan"]
for i in birthday.keys():
  if birthday[i]>maxv:
    maxv=birthday[i]

print(birthday)
print(f"Minimum no of birthday in a month are {minv} ")
print(f"Maximum no of birthday in a month are {maxv} ")
jan
dec
nov
feb
july
feb
aug
sep
june
oct
feb
jan

{'jan': 2, 'feb': 3, 'mar': 0, 'apr': 0, 'may': 1, 'june': 1, 'july': 1, 'aug': 1, 'sep': 1, 'oct': 1, 'nov': 1, 'dec': 1}
Minimum no of birthday in a month are 0 
Maximum no of birthday in a month are 3 
