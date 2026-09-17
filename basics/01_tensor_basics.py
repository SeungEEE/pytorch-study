"""PyTorch 텐서 기초 실습."""

import torch


def main() -> None:
    # 같은 코드를 실행할 때 동일한 난수가 나오도록 시드를 고정합니다.
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    y = torch.rand((2, 2), device=device)

    print(f"PyTorch 버전: {torch.__version__}")
    print(f"사용 장치: {device}")
    print(f"\nx =\n{x}")
    print(f"\ny =\n{y}")
    print(f"\nx + y =\n{x + y}")
    print(f"\nx @ y =\n{x @ y}")
    print(f"\nx의 모양: {x.shape}")


if __name__ == "__main__":
    main()
