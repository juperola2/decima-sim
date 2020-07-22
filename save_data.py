import csv
import numpy as np
import os


def save_data(time, env, scheme, c, dir):
    fct = "{}{}_jct.csv".format(dir, scheme)
    fgd="{}{}_geral_data.csv".format(dir, scheme)
    ex_f = os.path.exists(fct)
    with open(fct, 'a', newline='') as csvfile:
        swt = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
        if not ex_f:
            swt.writerow(['start_time', 'completion_time'])
        for jd in env.finished_job_dags:
            swt.writerow([jd.start_time, jd.completion_time])
    ex_f = os.path.exists(fgd)
    with open(fgd, 'a', newline='') as csvfile:
        swt = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
        if not ex_f:
            swt.writerow(['scheme', 'number_events', 'dur_total'])
        swt.writerow([scheme, c, time])

def save_jcts_ep(data, dir):
    file = "{}jcts_train.csv".format(dir)
    ex_f = os.path.exists(file)
    with open(file, 'a', newline='') as csvfile:
        swt = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
        if not ex_f:
            swt.writerow(['ep', 'jct_avg_per_agent'])
            for d in data:
                l = [d[0]]
                for jct in d[1]:
                    l.append(jct)
                swt.writerow(l)
