import penguins as pg

data_1d = pg.read(".", 1, 1)
data_2d = pg.read(".", 2, 1)

fig, axs = pg.subplots2d(1, 2)

data_1d.stage(axs[0])   # 1D data on the left
pg.mkplot(axs[0])
axs[0].set_title("1D data")

data_2d.stage(axs[1])   # 2D data on the right
pg.mkplot(axs[1])
axs[1].set_title("2D data")

fig.suptitle("Some NMR data with different dimensions")
pg.show()