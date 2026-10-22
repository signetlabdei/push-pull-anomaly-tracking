 #
 # This file is part of the Push-Pull Medium Access repository:
 # https://github.com/signetlabdei/push-pull-anomaly-tracking
 # Copyright (c) 2025:
 # Fabio Saggese (fabio.saggese@ing.unipi.it)
 # Federico Chiariotti (federico.chiariotti@unipd.it)
 #
 # This program is free software: you can redistribute it and/or modify
 # it under the terms of the GNU General Public License as published by
 # the Free Software Foundation, version 3.
 #
 # This program is distributed in the hope that it will be useful, but
 # WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 # General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License
 # along with this program. If not, see <http://www.gnu.org/licenses/>.
 #


import numpy as np
import math
import matplotlib.pyplot as plt

def bayes_collisions(P, A, c, s, activation):
    """Use Bayes' theorem to compute the probability of a given number of nodes attempting a transmission for a given outcome

    :param P: number of resources (slots) in the push subframe
    :param A: number of potential transmitting nodes
    :param c: number of collided slots
    :param s: number of successful slots
    :param activation: average activation probability

    :return: the belief distribution (NxM matrix)
    """
    p_cs_a = np.zeros(A + 1)
    p_a = np.zeros(A + 1)
    for a in range(A + 1):
        p_a[a] = np.power(activation, a) * np.power(1 - activation, A - a) * math.comb(A, a)
        if a >= 2 * c + s and p_a[a] > 1e-6:
            p_cs_a[a] = np.power(1 / P, s) * math.factorial(P) / math.factorial(P - s) * math.comb(a, s)
            p_cs_a[a] *= np.power(1 / P, 2 * c) * math.factorial(P - s) / math.factorial(P - s - c) * math.factorial(c) * math.comb(a - s, c) * math.comb(a - s - c, c) / np.power(2, c)
            p_cs_a[a] *= np.power(c / P, a - 2 * c - s) / np.power(3, a - 2 * c - s)
    p_a_cs = np.zeros(A + 1)
    p_cs = 0
    for a in range(A + 1):
        p_cs += p_cs_a[a] * p_a[a]
    if p_cs > 0:
        for a in range(A + 1):
            p_a_cs[a] = p_cs_a[a] * p_a[a] / p_cs
    if np.sum(p_a_cs) > 0:
        p_a_cs /= np.sum(p_a_cs)
    return p_a_cs

def bayes_collisions_errors(P, A, c, s, activation, epsilon):
    """Use Bayes' theorem to compute the probability of a given number of nodes attempting a transmission for a given outcome

    :param P: number of resources (slots) in the push subframe
    :param A: number of potential transmitting nodes
    :param c: number of collided slots
    :param s: number of successful slots
    :param activation: average activation probability
    :param epsilon: erasure probability

    :return: the belief distribution (NxM matrix)
    """
    p_cs_a = np.zeros(A + 1)
    p_a = np.zeros(A + 1)
    for a in range(A + 1):
        p_a[a] = np.power(activation, a) * np.power(1 - activation, A - a) * math.comb(A, a)
        for e in range(c + 1):
            p_e = math.factorial(s + e) / math.factorial(s) / math.factorial(e) * np.power(epsilon, e) * np.power(1 - epsilon, s)
            coll = c - e
            single = s + e
            if a >= 2 * coll + single and p_a[a] > 1e-6:
                p_cs_a_e = np.power(1 / P, single) * math.factorial(P) / math.factorial(P - single) * math.comb(a, single)
                p_cs_a_e *= np.power(1 / P, 2 * coll) * math.factorial(P - single) / math.factorial(P - single - coll) * math.factorial(coll) * math.comb(a - single, coll) * math.comb(a - single - coll, coll) / np.power(2, coll)
                p_cs_a_e *= np.power(coll / P, a - 2 * coll - single) / np.power(3, a - 2 * coll - single)
                p_cs_a[a] += p_cs_a_e * p_e
    p_a_cs = np.zeros(A + 1)
    p_cs = 0
    for a in range(A + 1):
        p_cs += p_cs_a[a] * p_a[a]
    if p_cs > 0:
        for a in range(A + 1):
            p_a_cs[a] = p_cs_a[a] * p_a[a] / p_cs
    if np.sum(p_a_cs) > 0:
        p_a_cs /= np.sum(p_a_cs)
    return p_a_cs



### MAIN SCRIPT ###
if __name__ == '__main__':
    rng = np.random.default_rng(0)
    # System parameters
    T = 1000 # Number of samples
    N = 100
    alpha = 3 / 10000
    max_aoii = 100
    P = 10
    threshold = 2
    epsilon = 0.02
    # Maximum possible age and belief distribution
    max_aoiis = np.zeros(N)
    max_aoiis[:10] = 4
    max_aoiis[10:20] = 3
    max_aoiis[21:40] = 2
    max_aoiis[41:70] = 1
    psi = np.zeros((N, int(np.max(max_aoiis) + 1)))
    for n in range(N):
        psi[n, int(max_aoiis[n])] = alpha
        i = int(max_aoiis[n] - 1)
        while i >= 0:
            psi[n, i] = alpha * np.power(1 - alpha, max_aoiis[n] - i)
            i -= 1
        psi[n,:] /= np.sum(psi[n, :])
    # Activation probability
    act_prob = np.zeros(N)
    for n in range(N):
        act_prob[n] = np.sum(psi[n, threshold + 1:])
    A = np.sum(act_prob > 0)
    activation = np.sum(act_prob) / A
    # Parameter choice
    c = 2
    s = 2
    active = np.zeros(T)
    idx = 0
    tot_run = 0
    while (idx < T):
        tx = 0
        tot_run += 1
        for n in range(N):
            tx += rng.random() < act_prob[n]
        if (tx >= s + c):
            slots = rng.integers(0, P, tx)
            succ = 0
            coll = 0
            for p in range(P):
                att = len(np.where(slots == p)[0])
                if (att == 1):
                    if (rng.random() > epsilon):
                        succ += 1
                    else:
                        coll += 1
                if (att > 1):
                    coll += 1
            if (coll == c and succ == s):
                active[idx] = tx
                idx += 1
    empirical_p = np.histogram(active, np.arange(-0.5,A + 0.6, 1))
    print(T / tot_run)
    p_a_cs = bayes_collisions(P, A, c, s, activation)
    p_a_cs_err = bayes_collisions_errors(P, A, c, s, activation, epsilon)
    plt.plot(range(A + 1), p_a_cs, label='Computed (no errors)')
    plt.plot(range(A + 1), p_a_cs_err, label='Computed (errors)')
    plt.plot(range(A + 1), empirical_p[0] / T, label='Empirical')
    plt.legend()
    plt.show()
