import pickle
import zlib



# 解压pkl文件
with open('data/001.pkl','rb') as f:
    byte_file=f.read()
byte_file=zlib.decompress(byte_file)
data= pickle.loads(byte_file)
a=1