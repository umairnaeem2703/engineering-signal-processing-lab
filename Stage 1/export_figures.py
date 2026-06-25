from pathlib import Path
import runpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIGURE_DIR = Path("latex") / "figures"

EXPECTED_FIGURES = {
    "exp_1.py": [
        "exp1_time.png",
        "exp1_frequency.png",
    ],
    "exp_2.py": [
        "exp2_stem_time.png",
        "exp2_line_time.png",
        "exp2_frequency.png",
    ],
    "exp_3.py": [
        "exp3_sigma_0_0_time.png",
        "exp3_sigma_0_0_frequency.png",
        "exp3_sigma_0_2_time.png",
        "exp3_sigma_0_2_frequency.png",
        "exp3_sigma_0_5_time.png",
        "exp3_sigma_0_5_frequency.png",
        "exp3_sigma_1_0_time.png",
        "exp3_sigma_1_0_frequency.png",
    ],
    "exp_4a.py": [
        "exp4a_T2_time.png",
        "exp4a_T2_frequency.png",
        "exp4a_T5_time.png",
        "exp4a_T5_frequency.png",
        "exp4a_T20_time.png",
        "exp4a_T20_frequency.png",
    ],
    "exp_4b.py": [
        "exp4b_T2_time.png",
        "exp4b_T2_frequency.png",
        "exp4b_T5_time.png",
        "exp4b_T5_frequency.png",
        "exp4b_T20_time.png",
        "exp4b_T20_frequency.png",
    ],
    "exp_5a.py": [
        "exp5a_time.png",
        "exp5a_frequency.png",
    ],
    "exp_5b.py": [
        "exp5b_time.png",
        "exp5b_frequency.png",
    ],
    "exp_5c.py": [
        "exp5c_window_comparison.png",
    ],
}


def export_script_figures(script_name, names):
    index = 0

    def save_current_figure(*args, **kwargs):
        nonlocal index
        if index >= len(names):
            raise RuntimeError(f"{script_name} produced more figures than expected")

        output_path = FIGURE_DIR / names[index]
        plt.gcf().savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"{script_name}: saved {output_path}")
        plt.close(plt.gcf())
        index += 1

    original_show = plt.show
    plt.show = save_current_figure
    try:
        runpy.run_path(script_name, run_name="__main__")
    finally:
        plt.show = original_show
        plt.close("all")

    if index != len(names):
        raise RuntimeError(
            f"{script_name} produced {index} figures; expected {len(names)}"
        )


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    for script_name, figure_names in EXPECTED_FIGURES.items():
        export_script_figures(script_name, figure_names)


if __name__ == "__main__":
    main()
