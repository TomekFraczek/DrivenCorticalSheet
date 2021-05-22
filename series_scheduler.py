from multiprocessing import Process

from main import run


if __name__=='__main__':

    sets = ('local_sync',)
    processes = []

    for s in sets:
       p = Process(target=run, args=(s,'model_config.json','plots',True))
       p.start()
       p.join()
       processes.append(p)
