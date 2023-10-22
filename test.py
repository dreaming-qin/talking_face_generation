from multiprocess import Pool
 

if __name__=='__main__':
    f_list = [1, 2, 3]
    
    def f(a):
        # Some unrelated code 
        print(a)
        return a

    with Pool(2) as pool:
        for a in pool.imap_unordered(f, f_list):
            # Some unrelated code
            pass
 