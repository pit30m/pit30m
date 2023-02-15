from itertools import zip_longest


def print_list_with_limit(lst, limit: int, logger=None) -> None:
    out = ""
    for entry in lst[:limit]:
        out += f"\t - {entry}\n"
    if len(lst) > limit:
        out += f"\t - ... and {len(lst) - limit} more."

    if logger is None:
        print(out)
    else:
        logger.info("%s", out)


def safe_zip(*args):
    """Pre Python 3.10 zip which raises if the lengths of the iterables are not equal.

    TODO(andrei): Python 3.10 was released in Oct 2021. We can drop support for it in, like, summer '23.
    See: https://peps.python.org/pep-0618/
    """
    sentinel = object()
    for tup in zip_longest(*args, fillvalue=sentinel):
        if sentinel in tup:
            raise ValueError("Iterables must be of equal length.")
        yield tup
