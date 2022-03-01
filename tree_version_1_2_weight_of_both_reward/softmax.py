import math

#a=[20,30,40,50,60,70]
# print(math.exp(0))
# print(math.exp(1))
# print(math.exp(30))
# a=[0,1,3,6,10,15,23,30]
# tempb=0
# for i in a:
#     tempb+=math.exp(i)
# for i in a:
#     res=math.exp(i)/tempb
#     print(res)
#
# print("---------------------------")

a=[0,1,2,4,8,14,21,29]
b=[]
tempb=0
for i in a:
    tempb+=i
for i in a:
    res=i/tempb
    b .append(res)
    print(res)
print(b)
a=[0,1,2,3,4,5,6,7]
b=[]
tempb=0
for i in a:
    tempb+=i
for i in a:
    res=i/tempb
    b .append(res)
    print(res)
print(b)
# tempb=0
# for i in b:
#     tempb+=math.exp(i)
# for i in b:
#     res=math.exp(i)/tempb
#     print(res)

print("---------------------------")
# a=[20,30,40,50,60,70]
# tempb=0
# for i in a:
#     tempb+=math.exp(i)
# for i in a:
#     res=math.exp(i)/tempb
#     print(res)