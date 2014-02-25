import sys

arg1 = sys.argv[1] #--> input file
arg2 = sys.argv[2] #--> output file

inputFiles = arg1.split(',')
files_list = open(arg2,"w") 
for inputFile in inputFiles:
    print "add %s" % inputFile
    files_list.write(inputFile+"\n")

files_list.close()
sys.exit(0)
