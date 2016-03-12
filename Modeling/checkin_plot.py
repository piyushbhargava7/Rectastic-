import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_checkin_b = train_checkin_df.join(train_b.select(train_b.stars,  train_b.business_id), train_checkin_df.b_id == train_b.business_id, 'inner')

checkin_summary_df= train_checkin_b.groupBy(train_checkin_b.stars).sum().toPandas().sort(['stars'])

checkin_summary_df_t = checkin_summary_df.transpose().drop(['sum(stars)','stars']).sort()

ll=[]
for idx,row in checkin_summary_df_t.iteritems():
    l = row.values.tolist()
    l.insert(0,idx)
    ll.append(l)


csfont = {'fontname':'Times New Roman'}
hfont = {'fontname':'Times New Roman'}
checkin_times = np.arange(0,168,1)
fig = plt.figure()
ax = fig.add_subplot(111)
# plt.plot(depth_trees, seg_1, color='b', label='Segment 1 - Known User, Known business')
# plt.plot(depth_trees, seg_2, color='g', label='Segment 2 - Unknown User, Known business')
# plt.plot(depth_trees, seg_3, color='r', label='Segment 3 - Known User, Unknown business')
# plt.plot(depth_trees, seg_4, color='y', label='Segment 4 - Known User, Known business in Test only')
# plt.plot(depth_trees, seg_5, color='m', label='Segment 5 - Unknown User, Known business in Test only')
# plt.plot(checkin_times, ll[1][1:169], color='b', label='Rating 1.5')
plt.plot(checkin_times, ll[2][1:169], color='k', label='Rating 2.0')
plt.plot(checkin_times, ll[3][1:169], color='m', label='Rating 2.5')
plt.plot(checkin_times, ll[4][1:169], color='y', label='Rating 3.0')
plt.plot(checkin_times, ll[5][1:169], color='c', label='Rating 3.5')
plt.plot(checkin_times, ll[6][1:169], color='r', label='Rating 4.0')
plt.plot(checkin_times, ll[7][1:169], color='g', label='Rating 4.5')
plt.plot(checkin_times, ll[8][1:169], color='b', label='Rating 5.0')
plt.title("Number of Checkins by Average Business Rating", fontsize=16, **csfont)
plt.xlabel("Weekly hours from Sunday Early Morning to Saturday Night", fontsize=14,**hfont)
plt.ylabel("Number of Checkins", fontsize=14, **hfont)
plt.legend(loc='upper left', fontsize=12,ncol=2)
# plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#           fancybox=True, shadow=True, ncol=5)
# plt.legend(loc='upper left', bbox_to_anchor=(0.8, 0.45),
#           ncol=1, fancybox=True, shadow=True)
plt.grid()
plt.savefig("2.pdf", format="pdf")
plt.show()

