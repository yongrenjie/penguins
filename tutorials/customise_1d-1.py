import penguins as pg
data_1d = pg.read(".", 1, 1)    # read in data
# >>>
data_1d.stage()                 # stage
pg.mkplot()                     # construct
pg.show()                       # display
