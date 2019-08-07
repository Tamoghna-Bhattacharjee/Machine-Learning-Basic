import pandas, random
from matplotlib import pyplot as plt
import seaborn as sns

df = pandas.read_csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10
total_reward = 0
ad_shown = []
number_of_times_reward_0 = [0] * d
number_of_times_reward_1 = [0] * d

for n in range(N):
    ad = 0
    max_beta = 0
    for j in range(d):
        beta = random.betavariate(number_of_times_reward_1[j] + 1, number_of_times_reward_0[j] + 1)
        if beta > max_beta:
            max_beta = beta
            ad = j
    ad_shown.append(ad)
    reward = df.iloc[n, ad]
    total_reward += reward
    if reward == 0:
        number_of_times_reward_0[ad] += 1
    else:
        number_of_times_reward_1[ad] += 1

print(ad_shown)
print(total_reward)
plt.hist(ad_shown)
plt.show()