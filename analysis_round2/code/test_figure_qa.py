"""Regression test for the round-two figure overlap detector."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_qa import check


def test_legend_over_xlabel_is_detected():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1], label="Series")
    ax.set_xlabel("Deliberately covered axis label")
    # Place the legend directly over the x-axis label.
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.canvas.draw()
    problems = check(fig, "deliberate_overlap.png")
    plt.close(fig)
    assert any("overlap" in problem for problem in problems), problems


if __name__ == "__main__":
    test_legend_over_xlabel_is_detected()
    print("PASS: deliberate legend/x-label overlap detected")
