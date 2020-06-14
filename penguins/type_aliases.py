from __future__ import annotations
from typing import Optional, Tuple, Union

OF = Optional[float]
OI = Optional[int]
OS = Optional[str]
TBounds1D = Tuple[OF, OF]
TBounds2D = Tuple[TBounds1D, TBounds1D]
TColors = Tuple[OS, OS]
TLevels = Union[Tuple[OF, OF, OI], float]
