import penguins as pg
data_2d = pg.read(".", 2, 1)
data_2d.stage()
pg.mkplot()
pg.show()