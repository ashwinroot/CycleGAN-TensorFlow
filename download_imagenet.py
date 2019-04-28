import urllib
import os

folder_name = "data/mango/"
synset_txt = "imagenet_links/mango.txt"

try:
    if(os.path.exists(folder_name)):
        os.rmdir(folder_name)
    os.mkdir(folder_name)
    os.mkdir("{}{}/".format(folder_name,"train"))
    os.mkdir("{}{}/".format(folder_name,"test"))
except OSError:
    print ("Creation of the directory %s failed" % folder_name)
    # exit()

with open(synset_txt,"r") as f:
    lines= f.readlines()
    test_size = int(len(lines)*0.2)
    print("test size",test_size)
    for i,line in enumerate(lines):
        if i<test_size:
            path = "{}test/n_{}.jpg".format(folder_name,i)
        else:
            path = "{}train/n_{}.jpg".format(folder_name,i)
        if i%500 == 0:
            print("Done : {}/{}".format(i,len(lines)))
        # path = "{}n_{}.jpg".format(folder_name,i)
        try:
            # print(line)
            urllib.request.urlretrieve(line, path)
        except Exception as e:
            print(str(e))
            print("Some problem for {}".format(i))
            pass
    print("done")