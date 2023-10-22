number = 42
number_str = str(number)
bytes_obj = bytes(number_str, 'utf-8')
with open('a.txt','wb') as f:
    f.write(bytes_obj)
print(bytes_obj)