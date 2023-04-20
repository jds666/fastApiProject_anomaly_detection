import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'Simhei'  # 显示中文

# matplotlib可视化结果
def matplotlib_show(x, y, ts, a, tl):
    """
    :param x: 散点图的横坐标
    :param y: 散点图的纵坐标
    :param ts: 条形图的纵坐标
    :param a: 条形图的横坐标
    :param tl:  图的标题
    :return:
    """
    # plt.rcParams['font.sans-serif'] = 'Simhei'  # 显示中文
    plt.figure(figsize=(16, 10), dpi=150)
    plt.scatter(x, y, edgecolors='b')
    plt.plot(ts, a)
    # plt.xticks(ts.values[::1000]) # 坐标轴每隔1000取一个
    plt.gcf().autofmt_xdate()  # 自适应X坐标
    plt.title(tl)
    plt.xlabel('时间')
    plt.ylabel('数值')
    plt.show()
