

f = open('test.txt', 'r')
f1 = open('test1.txt', 'w')

for line in f:
    line1 = line.rstrip()
    u1, u2, _ = line1.split()
    l = u1 + '\t' + u2 + '\n'
    f1.write(l)


