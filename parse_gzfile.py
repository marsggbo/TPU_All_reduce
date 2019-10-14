import json
import gzip
import glob
import numpy as np
import os

def parse(gzfile):
    with gzip.open(gzfile, 'rb') as f:
        data = json.load(f)
    tr = data['traceEvents']
    count = 0
    mean = 0.
    scale = int(gzfile.split('/')[0].split('scale')[1])
    size = 0.5*scale
    for t in tr:
        try:
            if t['name'] == 'cross-replica-sum':
                count += 1
                mean += t['dur']
            if count >= 5:
                break
        except Exception as e:
            print(gzfile, "Error!!")
            print(e)
            continue
    print(gzfile, mean/count, size/(mean/count)*1000**2/1024)

    return mean/count

def cp():
    tpu8_dirs = ['mnist_tpuv2n8_bs1024_scale{}'.format(i) for i in [2**j for j in range(12)]]
    tpu32_dirs = ['mnist_tpu32_bs4096_scale{}'.format(i) for i in [2**j for j in range(12)]]
    tpu128_dirs = ['mnist_tpu128_bs16384_scale{}'.format(i) for i in [2**j for j in range(12)]]
    tpu512_dirs = ['mnist_tpu512a_bs65536_scale{}'.format(i) for i in [2**j for j in range(12)]]
    dirs = tpu128_dirs + tpu512_dirs
    for d in dirs:
        if not os.path.exists(d):
            os.mkdir(d)
        os.system('gsutil cp gs://hkbuautoml/{}/plugins/profile/*/*.gz {}/'.format(d,d))

if __name__ == '__main__':
    logs = {}
    files = glob.glob('*/*gz')
    for file in files:
        name = file.split('/')[0]
        scale = int(name.split('scale')[1])
        model_size = scale*0.5
        time = parse(file)
        if name not in logs:
            logs[name] = {'time': [time], 'model_size': model_size}
        else:
            logs[name]['time'].append(time)
        
    for name in logs:
        model_size = logs[name]['model_size']
        mean_time = np.mean(logs[name]['time'])
        bandwidth = model_size * 1000 / (mean_time * 1024)
        print("{} model_size:{}MB mean_time:{}s bandwidth:{}GB/s".format(name, model_size, mean_time, bandwidth))
    
    with open('log.json', 'w') as f:
        json.dump(logs, f)

            