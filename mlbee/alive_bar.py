"""Customize alive_progress.alive_bar."""
from alive_progress import alive_bar as o_alive_bar


def alive_bar(*args, **kwargs):
    """Customize alive_progress.alive_bar.

    Default: total=1, length=3
    """
    # to prevent multiple values for keyword argument
    local_kw = dict(total=1, force_tty=True, length=3, title="diggin...")
    local_kw.update(**kwargs)
    if local_kw.get("length"):
        try:
            local_kw["length"] = int(local_kw.get("length"))
        except Exception:  # pylint: disable=bare-except
            local_kw["length"] = 3
        if local_kw.get("length") < 3:
            local_kw["length"] = 3  # or is invalid

    # update length for total > 3
    if local_kw["total"] > 3 and local_kw["length"] == 3:
        local_kw["length"] = local_kw["total"]

    return o_alive_bar(*args, **local_kw)
