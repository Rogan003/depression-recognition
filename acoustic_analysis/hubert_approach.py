import common

MODEL_NAME = "facebook/hubert-base-ls960"
LABEL = "HuBERT"

WIN_LENGTH_S = 5.0
HOP_LENGTH_S = 3.0


def main():
    common.run_acoustic_pipeline(
        MODEL_NAME, LABEL,
        win_length_s=WIN_LENGTH_S,
        hop_length_s=HOP_LENGTH_S,
    )


if __name__ == "__main__":
    main()
