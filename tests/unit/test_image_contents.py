"""What has to be inside the image, checked without building one.

A static read of the Dockerfile, which is a weaker check than running the container and is
worth having anyway: it catches the regression in one second in every CI run, where the
container check is a Phase 8 exit criterion and a `docker build` on a cold cache here takes
twenty minutes.

It exists because of a real gap. The runtime stage copies the virtualenv and nothing else, on
the reasoning that the package is installed into it -- correct for code, wrong for data. The
retrieval corpus is data. Without it in the image, `AGENTGATE_CORPUS_PATH` resolves to a
directory that is not there and every research branch fails, and no offline test could show
that, because the offline suite runs from a checkout where `corpus/` is simply present.

The general shape: **the tests run somewhere the application will not.** Anything the
application reads off the filesystem is a candidate for this failure, so the assertion below
is written against the corpus directory as a member of that category rather than as a
one-off.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentgate.config import Settings

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")

pytestmark = pytest.mark.usefixtures("isolated_env")


def runtime_stage() -> str:
    """Everything after the final ``FROM``. The builder stage does not ship."""
    return DOCKERFILE.rsplit("FROM ", 1)[-1]


def test_the_runtime_stage_is_the_one_being_checked() -> None:
    """Guards the tests below: a copy in the builder stage does not put anything in the image,
    so matching against the whole file would pass on a line that ships nothing."""
    stage = runtime_stage()

    assert "USER" in stage, "the final stage does not look like the runtime stage"
    assert len(stage) < len(DOCKERFILE), "there is only one stage; the split found nothing"


def test_the_corpus_ships_in_the_image() -> None:
    """Data the application reads at runtime has to be copied explicitly. Code does not, which
    is exactly why this one was missed."""
    assert "corpus/" in runtime_stage(), (
        "the retrieval corpus is not copied into the runtime stage. Every research branch will "
        "fail inside the container with CorpusUnavailableError, and no offline test will show "
        "it, because the suite runs from a checkout where corpus/ exists."
    )


def test_the_corpus_default_path_is_what_the_image_provides() -> None:
    """The copy and the default have to agree. Two plausible values -- `/app/corpus` in the
    image and `corpus` in configuration -- that resolve to the same place only because WORKDIR
    is `/app`, which is worth asserting rather than remembering."""
    default = Settings(_env_file=None, lane="fake").corpus_path  # type: ignore[call-arg]

    assert "WORKDIR /app" in runtime_stage()
    assert f"/app/{default.as_posix()}/" in runtime_stage()
