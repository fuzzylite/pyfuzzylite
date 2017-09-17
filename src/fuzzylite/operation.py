decimals = 3


def valid_name(name: str) -> str:
    result = ''.join([x for x in name if x in ("_", ".") or x.isalnum()])
    return result if result else "unnamed"


def str_(x: float) -> str:
    global decimals
    return ("{:.%sf}" % decimals).format(x)
