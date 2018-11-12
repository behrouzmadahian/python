import multiprocessing
import numpy as np
if __name__=='__main__':
    # using process and run completely independently:
    l2_grid = np.linspace(0, 5, 21)
    for l2Reg in l2_grid:
        if not os.path.exists('rollTraining/%s/predictions/l2-%.4f' % (res_folder, l2Reg)):
            os.makedirs('rollTraining/%s/predictions/l2-%.4f' % (res_folder, l2Reg))
        processes = []
        p = multiprocessing.Process(target=myparralelFunc,
                                    args=(l2Reg, train_dict, test_dict, trainFrac, startInds,
                                          slide_size, markets, transCost_dict))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # apply asyncronously
        l2_grid = np.linspace(0, 10, 101)
        pool = multiprocessing.Pool(processes=4)

        for l2 in l2_grid:
            p = pool.apply_async(myparralelFunc, args=(l2,))
        pool.close()
        pool.join()