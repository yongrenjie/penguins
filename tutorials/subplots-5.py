fig, axs = pg.subplots2d(2, 2)

contour_levels = [1e4, 3e4, 1e5, 3e5]
titles = ["Lots of noise", "Some noise",
          "Just a bit of noise", "No noise"]

for ax, level, title in zip(axs.flat, contour_levels, titles):
    data_2d.stage(ax=ax, levels=level)
    pg.mkplot(ax=ax, title=title)