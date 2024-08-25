import numpy as np
import matplotlib.pyplot as plt
import optim
import os
import time
import csv
import draw.convergence_rate
from joblib import Parallel, delayed

num_of_act = 10


def get_br_policy(bar_r1, bar_r2):
    return np.argmax(bar_r1), np.argmax(bar_r2)


def get_rm_policy(bar_r1, bar_r2):
    def regret_matching(r):
        copy_r = r.copy()
        copy_r[copy_r < 0] = 0
        if np.sum(copy_r) == 0:
            return np.ones(num_of_act) / num_of_act
        else:
            return copy_r / np.sum(copy_r)

    return regret_matching(bar_r1), regret_matching(bar_r2)


def no_regret_learning(u1, u2, itr, mode, log_interval):
    def get_epsilon(bar_sigma1, bar_sigma2):
        tmp2 = np.matmul(bar_sigma1, u2)
        tmp1 = np.matmul(u1, bar_sigma2)
        return np.max(tmp1) + np.max(tmp2)

    def save_epsilon(i_itr):
        tmp_epsilon = get_epsilon(bar_sigma1 / np.sum(bar_sigma1), bar_sigma2 / np.sum(bar_sigma2))
        result_dict['epsilon'].append(tmp_epsilon)
        result_dict['regret'].append((np.max(bar_r1) + np.max(bar_r2)) / overall_w)
        result_dict['itr'].append(i_itr)
        result_dict['now_time'].append(time.time())
        result_dict['overall_w'].append(overall_w)

    def get_min_change(max_q, max_v, a):
        max_num = np.max(max_q)
        gap = max_num - max_q
        change_num = max_v - max_v[a]
        change_ge_zero = np.where(change_num > 0.0)
        tmp_time = gap[change_ge_zero] // change_num[change_ge_zero] + 1
        if not list(tmp_time):
            return 99999999999999999999999.0
        else:
            return np.min(tmp_time)

    result_dict = {
        'epsilon'  : [],
        'regret'   : [],
        'itr'      : [],
        'overall_w': [],
        'now_time' : [],
    }
    bar_sigma1 = np.zeros(num_of_act)
    bar_sigma2 = np.zeros(num_of_act)

    bar_r1 = np.zeros(num_of_act)
    bar_r2 = np.zeros(num_of_act)

    overall_w = 0
    log_base = 10

    if 'FP' in mode.split():
        sigma1 = np.random.randint(num_of_act)
        sigma2 = np.random.randint(num_of_act)
    else:
        sigma1 = np.random.rand(num_of_act)
        sigma1 = sigma1 / np.sum(sigma1)
        sigma2 = np.random.rand(num_of_act)
        sigma2 = sigma2 / np.sum(sigma2)

    for i in range(1, itr + 1):
        if 'FP' in mode.split():
            q1 = u1[:, sigma2]
            q2 = u2[sigma1, :]
            r1 = q1 - q1[sigma1]
            r2 = q2 - q2[sigma2]
        else:
            if 'MC' in mode.split():
                tmp_sigma1 = sigma1.copy()
                sigma1 = np.zeros(num_of_act)
                sigma1[np.random.choice(num_of_act, p=tmp_sigma1)] = 1.0

                tmp_sigma2 = sigma2.copy()
                sigma2 = np.zeros(num_of_act)
                sigma2[np.random.choice(num_of_act, p=tmp_sigma2)] = 1.0

            q1 = np.matmul(u1, sigma2)
            q2 = np.matmul(sigma1, u2)
            r1 = (q1 - np.dot(sigma1, q1))
            r2 = (q2 - np.dot(sigma2, q2))

        if 'Liner' in mode.split():
            w = i
        elif 'C' in mode.split():
            w = 1
        elif 'DW' in mode.split():
            w1 = get_min_change(bar_r1, q1, sigma1)
            w2 = get_min_change(bar_r2, q2, sigma2)
            w = min(w2, w1)
            if w == 99999999999999999999999.0:
                print('没有结果')
                return {}
        elif 'G' in mode.split():
            if i == 1:
                w = 1
            else:
                tmp_bar_q = np.concatenate((bar_r1, bar_r2))
                tmp_r = np.concatenate((r1, r2))
                w, _ = optim.find_optimal_weight(tmp_bar_q, tmp_r, overall_w)
            if w == np.inf:
                w = 1
            if 'F' in mode.split():
                w = max(w, overall_w / (2 * i))
        else:
            pass
        if '+' in mode.split() or 'Predictive' in mode.split():
            bar_r1 += r1
            bar_r2 += r2
        else:
            bar_r1 += w * r1
            bar_r2 += w * r2

        if 'FP' in mode.split():
            bar_sigma1[sigma1] += w
            bar_sigma2[sigma2] += w
        else:
            bar_sigma1 += (w * sigma1)
            bar_sigma2 += (w * sigma2)

        overall_w += w

        if 'FP' in mode.split():
            sigma1, sigma2 = get_br_policy(bar_r1, bar_r2)
        else:
            sigma1, sigma2 = get_rm_policy(bar_r1, bar_r2)

        if '+' in mode.split():
            bar_r1[bar_r1 < 0] = 0
            bar_r2[bar_r2 < 0] = 0

        if log_base <= i:
            log_base *= log_interval
            save_epsilon(i)

    save_epsilon(itr)

    return result_dict


def train_sec(i_itr):
    for i_key in train_dict:
        print(i_key)
        u1 = np.random.randn(num_of_act, num_of_act)
        u2 = -u1

        result_file_path = ''.join(
            [
                logdir,
                '/',
                i_key,
                '/',
                str(i_itr),
            ]
        )
        os.makedirs(result_file_path)
        start_time = time.time()
        tmp_result_dict = no_regret_learning(u1, u2, 10000, i_key, 1.5)
        if not tmp_result_dict:
            return

        with open(result_file_path + '/epsilon.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            result_keys = list(tmp_result_dict.keys())
            writer.writerow(result_keys)
            for i_eps in range(len(tmp_result_dict['itr'])):
                tmp_write_data = []
                for result_key in result_keys:
                    if result_key == 'now_time':
                        tmp_write_data.append((tmp_result_dict[result_key][i_eps] - start_time) * 1000)
                    else:
                        tmp_write_data.append(tmp_result_dict[result_key][i_eps])
                writer.writerow(tmp_write_data)
    return


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)
    train_dict = [
        'C FP',
        'C RM',
        'Liner FP',
        'Liner RM',
        'DW FP',
        'Liner RM +',
        'Liner Predictive RM +',
        'MC G F RM',
        'MC G RM',
    ]

    now_path_str = os.getcwd()

    now_time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    logdir = ''.join(
        [
            now_path_str,
            '/logGFSPSampling/Matrix/',
            now_time_str,
        ]
    )
    num_of_core = 4
    num_of_train = 4
    ans_list = Parallel(n_jobs=num_of_core)(
        delayed(train_sec)(i_itr) for i_itr in range(num_of_train)
    )

    plt.figure(figsize=(32, 10), dpi=180)
    plt.subplot(121)
    draw.convergence_rate.plt_perfect_game_convergence_inline(
        'Iteration',
        logdir,
        is_x_log=True,
        x_label_index=2,
        y_label_index=0,
        x_label_name='Iteration',
        y_label_name='Epsilon'
    )
    plt.subplot(122)
    draw.convergence_rate.plt_perfect_game_convergence_inline(
        'Time',
        logdir,
        is_x_log=True,
        x_label_index=4,
        y_label_index=0,
        x_label_name='Time',
        y_label_name='Epsilon'
    )
    plt.savefig(logdir + '/pic.png')
