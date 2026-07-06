import argparse


def test_parser_creates_subcommands() -> None:
    from main import parser

    assert isinstance(parser, argparse.ArgumentParser)
    info_args = parser.parse_args(["info"])
    assert info_args.command == "info"
    train_args = parser.parse_args(["train", "--data-dir", "data/raw"])
    assert train_args.command == "train"
    assert train_args.data_dir == "data/raw"
    assert train_args.model_type == "waveform"
    enhance_args = parser.parse_args(
        ["enhance", "--checkpoint", "model.pt", "--input", "in.wav", "--output", "out.wav"]
    )
    assert enhance_args.command == "enhance"
    assert enhance_args.checkpoint == "model.pt"


def test_cmd_info(capsys) -> None:
    from arete.info import cmd_info

    cmd_info(argparse.Namespace(config=None))
    captured = capsys.readouterr()
    assert "WaveformUNet" in captured.out
    assert "STFTUNet" in captured.out
    assert "parameters" in captured.out.lower()
