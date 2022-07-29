import penguins as pg

ds = pg.read('.', 101)

ds.stage()
pg.mkplot()
pg.show()
