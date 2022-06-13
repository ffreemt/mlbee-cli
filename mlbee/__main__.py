"""Prep __main__ entry."""
# pylint: disable=invalid-name, too-many-locals, too-many-arguments, too-many-branches, too-many-statements, duplicate-code, import-outside-toplevel
from pathlib import Path
from textwrap import dedent
from typing import List, Optional

import logzero
import pendulum
import typer
from alive_progress import alive_bar
from icecream import ic
from icecream import install as ic_install
from logzero import logger
from set_loglevel import set_loglevel

from mlbee import __version__, mlbee

# from mlbee.cmat2html import cmat2html
# from mlbee.loadtext import loadtext
from mlbee.loadparas import loadtext
from mlbee.save_xlsx_tsv_csv import save_xlsx_tsv_csv
from mlbee.texts2pairs import texts2pairs

# from mlbee.text2lists import text2lists

logzero.loglevel(set_loglevel())

# logger.info(" loglevel: %s", _ or 20)

# logger.debug(" debug: %s", __file__)
# logger.info(" info: %s", __file__)

ic_install()
ic.configureOutput(
    includeContext=True,
    # outputFunction=logger.info,
    outputFunction=logger.debug,
)
ic.enable()

app = typer.Typer(
    name="mlbee",
    add_completion=False,
    help="de-en-bee aligner",
)

esp_min_samples_expl = dedent(
    """
    Larger esp or smaller min_samples will result in more aligned pairs but also more false positives (pairs falsely identified as candidates). On the other hand, smaller esp or larger min_samples values tend to miss `good` pairs."""
).strip()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(
            f"{app.info.name} v.{__version__} -- visit https://bumblebee.freeforums.net/thread/6/mlbee-cli-related or join qq group 316287378 to chat about {app.info.name}."
        )
        raise typer.Exit()


@app.command()
def main(
    files: List[str] = typer.Argument(
        ...,
        metavar="file1 [file2]...",
        help="files (absolute or relative paths) to be aligned; Two (any of 50+ languages) files ought to be provided; other options not yet implemented.",
    ),
    save_xlsx: bool = typer.Option(
        True,
        help="Save xlsx.",
    ),
    save_tsv: bool = typer.Option(
        True,
        help="Save tsv.",
    ),
    save_csv: bool = typer.Option(
        False,
        help="Save csv.",
    ),
    version: Optional[bool] = typer.Option(  # pylint: disable=(unused-argument
        None,
        "--version",
        "-v",
        "-V",
        help="Show version info and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
):
    """Align texts via huggingface, fast.

    e.g.

    * mlbee file1 file2  # xlsx and tsv

    * mlbee file1 file2 --no-save-tsv  # just xlsx

    * mlbee file1 file2 --no-save-xlsx --save-csv  # just csv
    """
    logger.info("Collecting inputs...")
    # test show-plot -p
    # logger.info(show_plot)

    logger.debug("files: %s", files)

    if not save_xlsx and not save_tsv and not save_csv:
        exiting = typer.style("exiting...", fg=typer.colors.RED, bold=True)
        typer.echo(
            "None of save-xlsx, save-tsv and save-csv is set to True: nothing to do, "
            f"{exiting}"
        )
        raise typer.Exit(code=0)

    if not files:
        typer.echo("Provide at least one file")
        raise typer.Exit()

    # if len(files) == 1 and not need_sep:
    # "If you only provide one file, you'll also
    # have to specify -s or --need-sep."
    if len(files) == 1:
        typer.echo("Missing one file")
        typer.echo("Try again...")
        raise typer.Exit()

    for file_ in files:
        if not Path(file_).is_file():
            typer.echo(f" [{file_}] is not a file or does not exist.")
            raise typer.Exit()

    # paired, two files and need_sep not set
    # if len(files) == 2 and not need_sep:
    if len(files) == 2:
        try:
            text1 = loadtext(files[0])
        except Exception as e:
            logger.error(e)
            raise
        try:
            text2 = loadtext(files[1])
        except Exception as e:
            logger.error(e)
            raise

        list1 = [elm.strip() for elm in text1.splitlines() if elm.strip()]
        list2 = [elm.strip() for elm in text2.splitlines() if elm.strip()]

        logger.debug("len1: %s, len2: %s", len(list1), len(list2))

    # other cases: 2 files + need_sep not set, 3 or or files
    else:  # assume mixed german/english, separate
        typer.echo(" Not yet implemented, stay tuned.")
        raise typer.Exit()

        list1 = []
        list2 = []
        text = ""
        for file in files:
            text = text + "\n" + loadtext(file)
        # list1, list2 = text2lists(text)

    # print estimated completion time
    len1 = len(list1)
    len2 = len(list2)
    len12 = len1 + len2
    time_min = 0.4 / 12
    time_max = 1 / 12
    time_av = .66 / 12
    time0 = len12 * time_min
    time1 = len12 * time_max
    eta = pendulum.now() + pendulum.duration(seconds=len12 * time_av)
    in_words0 = pendulum.duration(seconds=time0).in_words()
    in_words1 = pendulum.duration(seconds=time1).in_words()
    diff_for_humans = eta.diff_for_humans()
    dt_str = eta.to_datetime_string()
    timezone_name = eta.timezone_name
    _ = (
        f"Estimated time to complete: {in_words0} to  {in_words1}; "
        f"ETA: {diff_for_humans} ({dt_str} {timezone_name}) "
    )

    logger.info(_)

    try:
        logger.info("Delegating tasks to huggingface...")
        with alive_bar(total=1, force_tty=True, length=3, title="diggin...") as abar:
            aligned_pairs = texts2pairs(text1, text2)
    except Exception as e:
        typer.echo(e)
        raise typer.Exit()

    logger.debug("Proceed with saving files...")

    _ = [save_xlsx, save_tsv, save_csv]
    file_ext = [v1 for v0, v1 in zip(_, [".xlsx", ".tsv", ".csv"]) if v0]
    logger.debug("[save_xlsx, save_tsv, save_csv]: %s, file_ext: %s", _, file_ext)

    if not file_ext:  # nothing to do
        raise typer.Exit(code=0)

    logger.info("Saving %s", file_ext)

    _ = Path(files[0]).with_suffix("").as_posix() + "-ali"
    _ = Path(_)
    save_xlsx_tsv_csv(
        aligned_pairs,
        file_ext=file_ext,
        file=_,
    )

    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
