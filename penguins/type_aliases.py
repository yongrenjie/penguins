from typing import Optional, Tuple, Union

OF = Optional[float]
OI = Optional[int]
OS = Optional[str]
TBounds = Union[str, Tuple[OF, OF]]
TColors = Tuple[OS, OS]
TLevels = Union[Tuple[OF, OF, OI], float, int]
