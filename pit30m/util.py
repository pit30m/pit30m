
def print_list_with_limit(lst, limit: int) -> None:
    for entry in lst[:limit]:
        print(f"\t - {entry}")
    if len(lst) > limit:
        print(f"\t - ... and {len(lst) - limit} more.")
