# this is the local test version of the rodan job
# ignore this file

import sys, json
# from AomrMeiOutput import AomrMeiOutput

if __name__== "__main__":
    (tmp, inJSOMR, version) = sys.argv

    with open(inJSOMR, 'r') as file:
        jsomr = json.loads(file.read())

    print jsomr

    kwargs = {

    }

    print version

