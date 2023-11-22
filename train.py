import os



cmd='cd data2 && cat data.tar.* > data.tar && tar -xvf data.tar'
os.system(cmd)
print('finish cmd')