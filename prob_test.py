import numpy as np
import math

P = 8
a = 10
T = int(1e6)
p_a = 0.05

ct = 4
st = 0


s = np.zeros(T)
c = np.zeros(T)

for t in range(T):
    if (np.mod(t, 10000) == 0):
        print(t)
    choices = np.random.randint(1, P + 1, a)
    outcome = np.zeros(P)
    for p in range(1, P + 1):
        chosen = np.where(choices == p)[0]
        if (chosen.size != 0):
            if (chosen.size == 1):
                s[t] += 1
            else:
                c[t] += 1

p_cs = len(np.where(np.logical_and(s == st, c == ct))[0]) / T

print(p_cs)

if (a >= 2 * ct + st and P >= ct + st):
    p_cs_a = np.power(1 / P, st) * math.factorial(P) / math.factorial(P - st) * math.comb(a, st)
    print(p_cs_a)
    p_cs_a *= np.power(1 / P, 2 * ct) * math.factorial(P - st) / math.factorial(P - st - ct) * math.factorial(ct) * math.comb(a - st, ct) * math.comb(a - st - ct, ct) / np.power(2, ct)
    print(p_cs_a)
    p_cs_a *= np.power(ct / P, a - 2 * ct - st) / np.power(3, a - 2 * ct - st)
    print(p_cs_a)

