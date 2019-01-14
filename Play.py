import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_raw = pd.read_csv("game_plays.csv")
print(data_raw.info())
data_raw.head(5)
data0 = data_raw.sample(500)
data0.head(2)
data0.columns

data0.groupby("team_id_for").agg({'periodTime':'count'})

a = np.array(list(range(1,24))+[np.NAN]).reshape(2,3,4)
a
pd.DataFrame([tuple(list(x)+[val]) for x, val in np.ndenumerate(a)])
for x, val in np.ndenumerate(a):
    print(list(x),[val])

a = list(enumerate(list(range(1,5)) + [np.NAN]))
print((a))

a = np.repeat(0,15).reshape(3,5)
a = np.arange(0, 2*np.pi, 0.1)
b = np.sin(a)

plt.subplot(2,2,1)
plt.xlabel('aa').ylabel("bb")
## plt.ylabel('bb')
plt.plot(a,b,'r-', a**2, b**2, 'p-')
plt.show()


fig, (ax1,ax2) = plt.subplots(1,2)
my_plotter(ax1, a,b, {'marker':'x'})
ax.plot(a,b)(ax2, a,b, {'marker':'0'})
plt.show()

titanic = sns.load_dataset('titanic')
titanic.describe()
sns.catplot("class", "survived", "sex", data = titanic, kind = "bar", palette = "muted", legend = True)
