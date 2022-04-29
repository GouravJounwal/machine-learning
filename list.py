li=[1,2,3,4,5]
print(li)
[1, 2, 3, 4, 5]
li=[5,7,1,9,3,5]
li.sort()
print(li)
[1, 3, 5, 5, 7, 9]
li=[5,7,1,9,3,5]
print(sorted(li))
[1, 3, 5, 5, 7, 9]
li.append("amit")
print(li)
[5, 7, 1, 9, 3, 5, 'amit']
print(li[0])
li=["0827CI191008","Amit",5,50,90]
print(li)
0827CI191008
['0827CI191008', 'Amit', 5, 50, 90]
print(li.pop())
print(li[:])
90
['0827CI191008', 'Amit', 5, 50]
a=[1,2,3,4,5]
print(max(a))
print(min(a))
li.reverse()
print(li)
c=list(reversed(li))
print(c)
b=li[0:3]
print(b)
5
1
['0827CI191008', 'Amit', 5, 50]
[50, 5, 'Amit', '0827CI191008']
['0827CI191008', 'Amit', 5]
a.append(1)
print(a)

a.extend(["amit","patel"])
print(a)

print(a[: :-1])
[1, 2, 3, 4, 5, 1]
[1, 2, 3, 4, 5, 1, 'amit', 'patel']
['patel', 'amit', 1, 5, 4, 3, 2, 1]
t=(1,2,3,4)
print(t)

t1=(3,4,5,6)
t+t1

lis=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
print(lis[::2])
print(lis[::-2],"------")
print(lis[3::-1])
print(lis[12:0:-2])

t3=[i for i in t1 if i%2==0]
print(t3)
(1, 2, 3, 4)
[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
[20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0] ------
[3, 2, 1, 0]
[12, 10, 8, 6, 4, 2]
[4, 6]
