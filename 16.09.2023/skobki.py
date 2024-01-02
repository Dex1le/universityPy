# def main():

s = input()

s1 = s
if '[' or '{' or '(' or ')' or '}' or ']' in s:
    s1 = s1.replace('(', ' ( ')
    s1 = s1.replace('[', ' [ ')
    s1 = s1.replace(']', ' ] ')
    s1 = s1.replace('}', ' } ')
    s1 = s1.replace('{', ' { ')
    s1 = s1.replace(')', ' ) ')
s1 = s1.split()

a1 = s
k = 0
a1 = a1.replace('[', ' ')
a1 = a1.replace(']', ' ')
a1 = a1.replace('{', ' ')
a1 = a1.replace('}', ' ')
while '(' and ')' in a1:
    a1 = a1.replace('(', ':', 1)
    a1 = a1.replace(')', ';', 1)
    k += 1

a2 = s
k1 = 0
a2 = a2.replace(')', ' ')
a2 = a2.replace('(', ' ')
a2 = a2.replace('{', ' ')
a2 = a2.replace('}', ' ')
while '[' and ']' in a2:
    a2 = a2.replace('[', ':', 1)
    a2 = a2.replace(']', ';', 1)
    k1+= 1

a3 = s
k2 = 0
a3 = a3.replace('[', ' ')
a3 = a3.replace(']', ' ')
a3 = a3.replace(')', ' ')
a3 = a3.replace('(', ' ')
print(a3)
while '{' and '}' in a3:
    a3 = a3.replace('{', ':', 1)
    a3 = a3.replace('}', ';', 1)
    k2 += 1
    print(a3)

if ('(' or ')' in a1):
    print('False')
else:
    print('True')

