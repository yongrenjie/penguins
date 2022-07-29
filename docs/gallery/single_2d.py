import penguins as pg

ds = pg.read('.', 102) 

ds.stage(levels=7e3, f1_bounds="112..129", f2_bounds="7..9.3")
pg.mkplot()
pg.ymove()
pg.show()
