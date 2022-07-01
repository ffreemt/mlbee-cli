"""Prep __main__ entry."""
# pylint: disable=invalid-name, too-many-locals, too-many-arguments, too-many-branches, too-many-statements, duplicate-code, import-outside-toplevel, too-many-boolean-expressions
from pathlib import Path
from textwrap import dedent
from typing import List, Optional

import httpx
import logzero
import more_itertools as mit
import pendulum
import psutil
import typer
from about_time import about_time

# from alive_progress import alive_bar
from aset2pairs import aset2pairs

# from cmat2aset import cmat2aset
from fetch_radio_embed import fetch_radio_embed
from fetch_radio_cmat2aset import fetch_radio_cmat2aset
from icecream import ic
from icecream import install as ic_install
from logzero import logger
from seg_text import seg_text
from set_loglevel import set_loglevel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from mlbee import __version__

# from mlbee.cmat2html import cmat2html
# from mlbee.loadtext import loadtext
from mlbee.loadparas import loadtext
from mlbee.save_xlsx_tsv_csv import save_xlsx_tsv_csv
from mlbee.texts2pairs import texts2pairs
from mlbee.alive_bar import alive_bar

# from mlbee.text2lists import text2lists

api_url_forindo = "http://forindo.net:7860/api/predict"
api_url_local = "http://127.0.0.1:7860/api/predict"
api_url_hf_dev = "https://hf.space/embed/mikeee/radio-mlbee-dev/+/api/predict/"
api_url_hf = "https://hf.space/embed/mikeee/radio-mlbee/+/api/predict/"
api_url_default = api_url_hf
api_url_default = api_url_forindo

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
    help="a multilingual-text aligner using huggingface radio-mlbee api",
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
        help="files (absolute or relative paths) to be aligned; Two files  (in any of 50+ languages) ought to be provided; other options not yet implemented.",
    ),
    split_to_sents: bool = typer.Option(
        False,
        "--split-to-sents",
        "-s",
        is_flag=True,
        help="Split to sentences before aligning.",
    ),
    sep_mixed_text: bool = typer.Option(
        False,
        "--sep-mixed-text",
        "-m",
        is_flag=True,
        help="Separate mixed text to two texts before aligning.",
    ),
    flag: bool = typer.Option(
        True,
        "--flag",
        "-f",
        # "--remote_align",
        # "-r",
        # is_flag=True,
        help="If unset (flag=True), do everything on the remote API. If set (flag=False), fetch embeddings from remote but process locally because of timeout issue. Consider turn this on for huggingface, e.g. -f -a hf",
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
    api_url: str = typer.Option(
        api_url_default, "--api-url", "-a", help=f"URL of the api to use. shortcuts: local={api_url_local}, hf={api_url_hf}, forindo={api_url_forindo}."
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
    """Align multilingual (50+ pairs) texts via huggingface, fast, and top-notch.

    e.g.

    * mlbee file1 file2  # xlsx and tsv

    * mlbee file1 file2 -a local  # use api at local radio_mlbee server

    * mlbee file1 file2 -s # split to sents and align

    * mlbee file1 -m  # separate mixed language texts and align

    * mlbee file1 -m -s  # separate mixed language texts, split to sents and align

    * mlbee file1 file2 --no-save-tsv  # just xlsx

    * mlbee file1 file2 --no-save-xlsx --save-csv  # just csv
    """
    if api_url.lower().startswith("local"):
        api_url = api_url_local
    if api_url.lower().startswith("forindo"):
        api_url = api_url_forindo
    if api_url.lower().startswith("hf") or api_url.startswith("huggingface"):
        api_url = api_url_hf

    logger.debug("files: %s", files)
    _ = [
        "api_url",
        "files",
        "split_to_sents",
        "sep_mixed_text",
        "flag",
        "save_xlsx",
        "save_tsv",
        "save_csv",
    ]
    options = [
        api_url,
        files,
        split_to_sents,
        sep_mixed_text,
        flag,
        save_xlsx,
        save_tsv,
        save_csv,
    ]
    logger.info("options: %s", dict(zip(_, options)))

    # raise typer.Exit(code=0)

    logger.info("Probing the api-url: %s", api_url)
    with about_time() as t:
        data = ["a", "b", False, False]
        try:
            _ = httpx.post(api_url, json={"data": data})
            _.raise_for_status()
        except Exception as exc:
            logger.error("%s", exc)
            logger.info(" Looks like %s is not available...", api_url)
            _ = f" Looks like {api_url} is not available...That's all we know."
            _ = typer.style(
                _,
                fg=typer.colors.WHITE,
                bg=typer.colors.RED,
                bold=True,
            )
            typer.echo(_)
            raise typer.Exit(code=1)

    logger.debug(" %s is alive, response time about %s", api_url, t.duration_human)
    rtt = typer.style(
        t.duration_human,
        fg=typer.colors.WHITE,
        bg=typer.colors.YELLOW,
        bold=True,
    )

    typer.echo(f"\t{api_url} is alive\n\t response time about {rtt}")

    logger.info("Collecting inputs...")

    # check save_xlsx, save_tsv, save_csv, exit early if none is True
    _ = [save_xlsx, save_tsv, save_csv]
    file_ext = [v1 for v0, v1 in zip(_, [".xlsx", ".tsv", ".csv"]) if v0]
    logger.debug("[save_xlsx, save_tsv, save_csv]: %s, file_ext: %s", _, file_ext)
    if not file_ext:  # nothing to do
        # early exit
        typer.secho(
            "\tNothing to do... turn on at least one of --save-xlsx --save-tsv --save-csv and try again.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit(code=0)

    if sep_mixed_text:
        try:
            from sep_text import sep_text
        except ModuleNotFoundError as exc:
            msg1 = typer.style("[sep-text]", fg=typer.colors.WHITE, bg=typer.colors.RED)
            msg2 = typer.style(
                "pip install sep-text", fg=typer.colors.WHITE, bg=typer.colors.RED, bold=True,
            )
            typer.echo(
                "To use this option to need to install sep-text, "
                f"either use 'pip install mlbee{msg1}' or "
                f"'{msg2}' separately."
            )
            raise typer.Exit() from exc

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
    # have to specify -s or --need-sep/sep-mixed-text."
    if len(files) == 1 and not sep_mixed_text:
        mixed = typer.style(
            "--sep-mixed-text", fg=typer.colors.WHITE, bg=typer.colors.RED, bold=True,
        )
        typer.echo(
            "If you only provide one file, you'll have to " f"use the {mixed} switch."
        )
        typer.echo("Try again...")
        raise typer.Exit()

    for file_ in files:
        if not Path(file_).is_file():
            typer.echo(f" [{file_}] is not a file or does not exist.")
            raise typer.Exit()

    # paired, two files and sep-mixed-text not set
    # if len(files) == 2 and not need_sep:
    if len(files) == 2 and not sep_mixed_text:
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

    # other cases: 2 files + sep_mixed_text/need_sep not set,
    # 3 or or files
    else:  # sep_mixed_text is True or len(files) != 2
        list1 = []
        list2 = []
        text = ""
        for file in files:
            text = text + "\n" + loadtext(file)

        # list1, list2 = text2lists(text)
        res = sep_text(text)  # type: ignore  # inline import

        assert (
            len(res) <= 2
        ), "This should not happen, something horrible must have happened..."

        if len(res) == 1:
            text1 = res[0]
            text2 = ""
            list1 = [_ for _ in res[0] if _.strip()]
            list2 = []
        else:
            text1, text2 = res  # pylint: disable=unbalanced-tuple-unpacking
            list1 = [_ for _ in text1 if _.strip()]
            list2 = [_ for _ in text2 if _.strip()]

    _ = """  # processed at hf
    if split_to_sents:
        list1 = seg_text(list1)
        list2 = seg_text(list2)
        text1 = "\n".join(list1)  # these will be sent to hf
        text2 = "\n".join(list2)
    # """

    len1 = len(list1)
    len2 = len(list2)
    len12 = len1 + len2

    # checking both list1 and list2 not empty
    if not (len1 and len2):
        logger.warning(" One or both texts are empty...")
        logger.info("Sample text1: %s..., text2: %s...", text1[:100], text2[:100])
        typer.secho(
            typer.style(
                "\tNothing to do, man...",
                bg=typer.colors.WHITE,
                fg=typer.colors.RED,
                bold=True,
            )
        )
        raise typer.Exit(code=1)

    # ########################################## branch
    # if flag is set, do everything  on remote api
    # if flag is not set, do everything on remote api for short batches
    # (hf timeout 60 secs)
    if flag or (
        len12 < 2000 and not split_to_sents or len12 < 600 and len12 < 2000 and not split_to_sents or len12 < 600 and split_to_sents
    ):
        logger.info("Doing everything on remote server... sit tight.")

        if split_to_sents:
            blocks = f" est. {3.5 * len12} sents"  # estimate 3.5x for sents
        else:
            blocks = f"{len12} paras"
        # print estimated completion time
        factor = 6.6
        time_min = 0.4 / 12 / factor
        time_max = 1 / 12 / factor
        time_av = 0.66 / 12 / factor
        time0 = len12 * time_min
        time1 = len12 * time_max
        eta = pendulum.now() + pendulum.duration(seconds=len12 * time_av)
        in_words0 = pendulum.duration(seconds=time0).in_words()
        in_words1 = pendulum.duration(seconds=time1).in_words()
        diff_for_humans = eta.diff_for_humans()
        dt_str = eta.to_datetime_string()
        timezone_name = eta.timezone_name
        _ = (
            f" tot: {len1} + {len2}, {len12} \n"
            f"Estimated time to complete: {in_words0} to  {in_words1}; "
            f"ETA: {diff_for_humans} ({dt_str} {timezone_name}) "
            "-- apparently only valid if the number of blocks is not too large, say, smaller than 4000"
        )

        logger.info(_)

        try:
            logger.info("Delegating tasks to remote API %s...", api_url)
            with alive_bar(
                total=1, force_tty=True, length=3, title="diggin..."
            ) as abar:
                aligned_pairs = texts2pairs(
                    text1,
                    text2,
                    split_to_sents=split_to_sents,
                    url=api_url,
                )
                abar()  # pylint: disable=not-callable
                print("")
        except Exception as e:
            typer.echo(e)
            raise typer.Exit()

    else:
        # flag always False at this point
        assert not flag, f" flag: {flag}, not supposed to happen."
        # ########################################## branch
        # long batch, get cmat from hf or other api
        # process locally
        # logger.info("Mainly on-premise, remote api does the heavy lifting ")

        # aligned_pairs = ""

        if split_to_sents:
            with alive_bar(total=2, title="\t splitting to sents on-premise") as abar:
                try:
                    list1 = seg_text(list1)
                except Exception as exc:
                    logger.error(exc)
                    logger.warning("Can't seem to split to sents. We leave as is.")
                abar()  # pylint: disable=not-callable
                try:
                    list2 = seg_text(list2)
                except Exception as exc:
                    logger.error(exc)
                    logger.warning("Can't seem to split to sents. We leave as is.")
                abar()  # pylint: disable=not-callable

        len1 = len(list1)
        len2 = len(list2)
        len12 = len1 + len2

        bsize = 2000
        tot = len1 // bsize + bool(len1 % bsize)
        tot += len2 // bsize + bool(len2 % bsize)

        block_name = "paras"
        if split_to_sents:
            block_name = "sents"
        logger.info("Fetch embeddings for %s + %s = %s %s", len1, len2, len12, block_name)

        vec1 = []
        vec2 = []

        _ = """
        with tqdm(total=tot, desc="  fetching embeddings", leave=True) as pbar:
            for chunk in mit.chunked(list1, bsize):
                try:
                    vec = fetch_radio_embed("\n".join(chunk))
                except Exception as exc:
                    logger.error(exc)
                    raise
                vec1.extend(vec)
                pbar.update()
            for chunk in mit.chunked(list2, bsize):
                try:
                    vec = fetch_radio_embed("\n".join(chunk))
                except Exception as exc:
                    logger.error(exc)
                    raise
                vec2.extend(vec)
                pbar.update()
        # """

        vec12 = []
        for chunk in tqdm(
            mit.chunked(list1 + list2, bsize),
            total=len12 // bsize + bool(len12 % bsize),
            desc="  fetching embeddings",
            leave=True,
        ):
            try:
                vec = fetch_radio_embed("\n".join(chunk))
            except Exception as exc:
                logger.error(exc)
                raise
            vec12.extend(vec)
        vec1 = vec12[:len(list1)]
        vec2 = vec12[len(list1):]

        len1 = len(vec1)
        len2 = len(vec2)

        # with alive_bar(title="diggin cmat..."):
        avail_memo = psutil.virtual_memory().available  # bits, / 1024**2 -> M
        req_memo = len1 * len2 * 8  # 8 bytes: float64

        avail_memo_ = round(avail_memo / 1024 ** 2 / 8, 2)  # /8->bytes
        req_memo_ = round(req_memo / 1024 ** 2, 2)
        logger.info("available memory: %s M, required memory: %s M", avail_memo_, req_memo_)

        if avail_memo_ < req_memo_:
            logger.warning("\n\t You are running low on RAM, expect OOM.")

        try:
            # note the order vec2, vec1
            cmat = cosine_similarity(vec2, vec1)
        except Exception as exc:
            logger.error("%s, exiting...", exc)
            if "unable to allocate" in str(exc).lower():
                logger.info("Looks like you need to get more RAM, sorry man.")
            raise typer.Exit(code=1)

        try:
            cmat = cmat.tolist()
        except Exception as exc:
            logger.error(exc)
            # logger.error("")
            raise typer.Exit(1)

        with alive_bar(title="diggin aset...") as abar:
            # aset = cmat2aset(cmat)
            try:
                aset = fetch_radio_cmat2aset(cmat)
            except Exception as exc:
                logger.error(exc)
                raise typer.Exit(1)

            logger.debug("aset[:10]: %s", aset[:10])
            logger.debug("")
            abar()  # pylint: disable=not-callable

        # with alive_bar(title="diggin aligned_pairs...") as abar:
        aligned_pairs = aset2pairs(list1, list2, aset)
        logger.debug("aligned_pairs[:10]: %s", aligned_pairs[:10])
        # abar()

    # ###### common part: saving
    logger.debug("Proceed with saving files...")

    # file_ext defined in early stage for early exit
    logger.info("Saving %s", file_ext)

    _ = Path(files[0]).with_suffix("").as_posix() + "-ali"
    if split_to_sents:
        _ = f"{_}-sents"
    _ = Path(_)
    save_xlsx_tsv_csv(
        aligned_pairs,
        file_ext=file_ext,
        file=_,
    )

    raise typer.Exit(code=0)


if __name__ == "__main__":
    try:
        app()
    except Exception as _:
        logger.error(_)
