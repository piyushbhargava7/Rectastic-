import matplotlib.pyplot as plt
csfont = {'fontname':'Times New Roman'}
hfont = {'fontname':'Times New Roman'}
depth_trees = [2,4,6,8,10,12, 14, 16]
seg_1 = [1.175171, 1.039497,0.969214, 0.944835, 0.933909, 0.935836, 0.942030, 0.94910]
seg_2 = [1.326541, 1.224030,1.181784, 1.176044, 1.175161, 1.182987, 1.194052, 1.206677]
seg_3 = [1.329311, 1.233656,1.200014, 1.180983, 1.179031, 1.183647, 1.194079, 1.216863]
seg_4 = [1.456897, 1.433507,1.412850,1.388784, 1.366395, 1.346992, 1.32989, 1.322268]
seg_5 = [1.459687, 1.436279,1.413778,1.390908,1.370173, 1.352678, 1.331665, 1.320540]
fig = plt.figure()
ax = fig.add_subplot(111)
# plt.plot(depth_trees, seg_1, color='b', label='Segment 1 - Known User, Known business')
# plt.plot(depth_trees, seg_2, color='g', label='Segment 2 - Unknown User, Known business')
# plt.plot(depth_trees, seg_3, color='r', label='Segment 3 - Known User, Unknown business')
# plt.plot(depth_trees, seg_4, color='y', label='Segment 4 - Known User, Known business in Test only')
# plt.plot(depth_trees, seg_5, color='m', label='Segment 5 - Unknown User, Known business in Test only')
plt.plot(depth_trees, seg_1, color='b', label='Segment 1')
plt.plot(depth_trees, seg_2, color='g', label='Segment 2')
plt.plot(depth_trees, seg_3, color='r', label='Segment 3')
plt.plot(depth_trees, seg_4, color='y', label='Segment 4')
plt.plot(depth_trees, seg_5, color='m', label='Segment 5')
plt.title("Cross Validation across segments", fontsize=16, **csfont)
plt.xlabel("Maximum depth of trees",fontsize=14,**hfont)
plt.ylabel("MSE - Mean Squared Error",fontsize=14,**hfont)
# plt.legend(loc='right', fontsize=12)
# plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#           fancybox=True, shadow=True, ncol=5)
plt.legend(loc='upper center', bbox_to_anchor=(0.85, 0.45),fontsize=12,
          ncol=1, fancybox=True, shadow=True)
plt.grid()
plt.savefig("1.png", format="png")
plt.show()


# 2


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
plt.savefig("2.png", format="png")
plt.show()



from matplotlib import pyplot as plt
import numpy as np
from matplotlib_venn import venn3, venn3_circles
plt.figure(figsize=(4,4))
v = venn3(subsets=(1, 1, 1, 1, 1, 1, 1), set_labels = ('A', 'B', 'C'))
v.get_patch_by_id('100').set_alpha(1.0)
v.get_patch_by_id('100').set_color('white')
v.get_label_by_id('100').set_text('Unknown')
v.get_label_by_id('A').set_text('Set "A"')
c = venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), linestyle='dashed')
c[0].set_lw(1.0)
c[0].set_ls('dotted')
plt.title("Sample Venn diagram")
plt.annotate('Unknown set', xy=v.get_label_by_id('100').get_position() - np.array([0, 0.05]), xytext=(-70,-70),
             ha='center', textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',color='gray'))
plt.show()


def palidrome(num):
    n=num
    rev = 0
    while (num > 0):
          dig = num % 10
          rev = rev * 10 + dig
          num = num / 10
    return n==rev

palidrome(1131)