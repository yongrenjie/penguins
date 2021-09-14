import penguins as pg

data_1d = pg.read(".", 1, 1)
data_1d.stage()
fig, ax = pg.mkplot()  # faster: pg.mkplot(xlabel="My x label")

ax.set_xlabel("My x label")
pg.show()