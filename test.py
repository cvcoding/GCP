a = 100

def testA(a):
    b=a
    global a
    a = b

    print(a)


def testB(a):
	print(a)

# print(a) # 100
testA(a)
testB(a)

# print(testA(a)) # 100
# print(testB(a)) # 200
# print(testA()) # 100
