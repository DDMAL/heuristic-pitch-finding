## NON Rodan version of this job
## all function names will be identical to Aomr functions

import sys



def run(inputs, outputs):

        return True

  


if __name__== "__main__":
    (tmp, inCC, inMiyao) = sys.argv

    # open files to be read, and output file
    fCC = open(inCC, 'r')
    fMiyao = open(inMiyao, 'r')
    fXML = open('CF-011/hpf.xml', 'w')

    CC = fCC.read()


    fXML.write(CC)



    # close files
    fCC.close()
    fMiyao.close()
    fXML.close()
    print "file written."




