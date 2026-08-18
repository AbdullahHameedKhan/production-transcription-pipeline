"""Command line entrypoint. Satisfies the script requirement, shares the pipeline core."""
from __future__ import annotations

from pathlib import Path

import typer

from .config import get_settings
from .exporters import EXPORTERS
from .pipeline import transcribe_file

app = typer.Typer(add_completion=False, help="Audio transcription pipeline.")


@app.command()
def transcribe(
    audio: Path = typer.Argument(..., exists=True, readable=True),
    fmt: str = typer.Option("json", "--format", "-f", help="json | srt | vtt | text"),
    output: Path = typer.Option(None, "--output", "-o", help="Write here instead of stdout."),
    model: str = typer.Option(None, help="Override the whisper model size."),
    language: str = typer.Option(None, help="Force a language, else auto detect."),
) -> None:
    if fmt not in EXPORTERS:
        raise typer.BadParameter(f"format must be one of {', '.join(EXPORTERS)}")

    from .asr import build_engine  # deferred so --help stays instant

    settings = get_settings()
    if model:
        settings = settings.model_copy(update={"whisper_model": model})
    if language:
        settings = settings.model_copy(update={"language": language})

    engine = build_engine(settings)
    result = transcribe_file(audio, settings=settings, engine=engine)
    rendered = EXPORTERS[fmt](result.transcript)

    if output:
        output.write_text(rendered, encoding="utf-8")
        typer.echo(f"wrote {output}")
    else:
        typer.echo(rendered)


if __name__ == "__main__":
    app()