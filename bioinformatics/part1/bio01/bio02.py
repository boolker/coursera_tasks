__author__ = 'admin'


with open ("c:\\WinPython\\input2.txt", "r") as myfile:
    data=myfile.read().replace('\n', '')

print data

datalen = len(data);
print datalen

dict = {"A":"T","C":"G","T":"A","G":"C"};
result = ""
i = 0

while i <= (datalen - 1):
    letter = data[datalen - 1 - i]
    pair = dict.get(letter,0)
    if pair != 0:
      result += pair

    i = i+1

print result
