import penguins as pg
hsqc_dataset = pg.read("tests/data/pt2", 4, 1)
hsqc_dataset.stage(f1_bounds="10..141",
                   f2_bounds="0.5..8.5",
                   colors=("seagreen", "hotpink"),
                   levels=2e4)
pg.mkplot(title="An example HSQC")
pg.show()
