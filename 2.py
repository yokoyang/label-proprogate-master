f2 = open("2.txt", "r")
X2 = list()
while True:
    line = f2.readline()
    if line:
        pass  # do something here
        line = line.strip()
        # print line
        X2.append(line)
    else:
        break
f2.close()
print len(X2)
