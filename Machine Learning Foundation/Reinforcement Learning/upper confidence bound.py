import pandas
from matplotlib import pyplot as plt
from math import sqrt, log2

df = pandas.read_csv('Ads_CTR_Optimisation.csv')

N = 10000
d = 10
ads_shown_to_each_user = []
no_times_ad_shown = [0] * d
sum_of_reward = [0] * d
total_reward = 0

for n in range(N):
    ad = 0
    max_bound = 0
    for j in range(d):
        if no_times_ad_shown[j] > 0:
            average_reward = sum_of_reward[j] / no_times_ad_shown[j]
            delta = sqrt(1.5 * log2(n) / no_times_ad_shown[j])
            upper_bound = average_reward + delta
        else:
            upper_bound = float('inf')
        if upper_bound > max_bound:
            max_bound = upper_bound
            ad = j

    ads_shown_to_each_user.append(ad)
    no_times_ad_shown[ad] += 1
    sum_of_reward[ad] += df.iloc[n, ad]
    total_reward += df.iloc[n, ad]


print(ads_shown_to_each_user)
print(total_reward)
print(no_times_ad_shown)
plt.hist(ads_shown_to_each_user)
plt.show()
