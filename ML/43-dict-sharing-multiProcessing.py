import multiprocessing
def myF(shared_dict,i ):
    # do some work
    shared_dict[i] = i

if __name__ == '__main__':
    mgr = multiprocessing.Manager()
    d = mgr.dict()
    proceses=[]
    for i in range(10):
        p = multiprocessing.Process(target=myF, args=(d, i))
        p.start()
        proceses.append(p)
    for p in proceses:
        p.join()
    print(d)
    for item in d.keys():
        print(d[item])