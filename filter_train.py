import sys

file = sys.argv[1]
file1 = sys.argv[2]

f = open(file, 'r')
f1 = open(file1, 'w')


for line in f:
    line1 = line.rstrip()
    u, v, l = map(lambda x:int(x), line1.split())
    if l == 1:
        line2 = str(u) + '\t' + str(v)  + '\n'
        f1.write(line2)


