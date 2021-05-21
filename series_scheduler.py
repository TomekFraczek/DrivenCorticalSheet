from multiprocessing import Process

from main import main


if __name__=='__main__':
    
    sets = ('test_set','test_set_bifn')
    processes = []

    for s in sets:
       p = Process(target=main, args=(s,'--plot'))
       p.start()
       p.join()
       processes.append(p)
