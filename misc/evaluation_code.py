# -*- coding: utf-8 -*-

import os
import sys
import math

#MAE = lambda

errorcode = 'WA'

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.maeSum = 0
        self.mseSum = 0
        self.naeSum = 0
        self.count = 0
        self.naecount = 0

    def update(self, mae, mse, nae):
        
        self.maeSum += mae
        self.mseSum += mse
        if nae >= 0:
            self.naeSum += nae
            self.naecount += 1
        self.count += 1
    
    def output(self):
        if self.count > 0:
            mae = self.maeSum / self.count
            mse = math.sqrt(self.mseSum / self.count)
        else:
            mae, mse = -1, -1
        nae = self.naeSum / self.naecount if self.naecount > 0 else  -1
        return mae, mse, nae
    
    def dictout(self):
        mae, mse, nae = self.output()
        return dict(
            mae = mae,
            mse = mse,
            nae = nae
        )


def readoutput(outtxt):
    output = {}
    with open(outtxt) as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            if len(line) == 2:
                idx, score = int(line[0]), float(line[1])
                output[idx] = score
    return output

def readtarget(tartxt):
    target = {}
    with open(tartxt) as f:
        for line in f.readlines():
            line = line.strip().split(' ')
            if len(line) == 4:
                idx, illum, level = map(int, line[:3])
                score = float(line[3])
                target[idx] = dict(
                    illum = illum,
                    level = level,
                    gt_count = score
                )
    return target

def judge(outtxt, tartxt):
    output = readoutput(outtxt)
    target = readtarget(tartxt)
    for key in target.keys():
        if key in output:
            target[key]["pd_count"] = output[key]
        else:
            return errorcode
    
    totalJudger = AverageMeter()
    levelJudger = [AverageMeter() for _ in range(5)]
    illumJudger = [AverageMeter() for _ in range(4)]

    for _, score in target.items():
        # get data
        illum = score['illum']
        level = score['level']
        gt_count = score['gt_count']
        pd_count = score['pd_count']

        # process
        mae = abs(pd_count - gt_count)
        mse = mae ** 2
        nae = mae / gt_count if gt_count > 0 else -1

        # save
        totalJudger.update(mae, mse, nae)
        levelJudger[level].update(mae, mse, nae)
        illumJudger[illum].update(mae, mse, nae)
    
    outputdict = {
        'overall': totalJudger.dictout(),
        'levels': [judger.dictout() for judger in levelJudger],
        'illums': [judger.dictout() for judger in illumJudger],
    }
    outputdict['mmae'] = dict(
        mmae_level = sum(result['mae'] for result in outputdict['levels']) / len(outputdict['levels']),
        mmae_illum = sum(result['mae'] for result in outputdict['illums']) / len(outputdict['illums'])
    )

    return outputdict


if __name__ == '__main__':
    target = {}
    if len(sys.argv) != 3:
        print(errorcode)
    print(judge(sys.argv[1], sys.argv[2]))