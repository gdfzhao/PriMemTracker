import numpy as np
import multiprocessing


def parallel_work(datas, worker, *args):
    n_cpu = multiprocessing.cpu_count()
    print('n_cpu', n_cpu)
    num_data = len(datas)
    stride = int(np.ceil(num_data / n_cpu))

    processes = []
    for i in range(n_cpu):
        end = min((i + 1) * stride, num_data)
        worker_data = datas[i * stride:end]

        process = multiprocessing.Process(
            target=worker,
            args=(worker_data, *args),
        )
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
