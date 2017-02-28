TEST_TXT = "2"
params={'1':2,
#this
'3':4
}
for i in range(10):
	f = open('result'+TEST_TXT+'.txt','a')

	f.write(str(i)+' '+'hi\n')
	f.write(str(params))
	f.close()
