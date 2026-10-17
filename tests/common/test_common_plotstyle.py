"""Smoke test for common.plotstyle: serif body font, vector PDF output."""
import matplotlib

from common import plotstyle


def test_style_is_applied_and_saves_a_vector_pdf(tmp_path):
    font = plotstyle.apply()
    assert matplotlib.rcParams["font.size"] == 10
    assert matplotlib.rcParams["mathtext.fontset"] == "stix"
    assert font in ("Nimbus Roman", "TeX Gyre Termes", "serif")
    fig, ax = plotstyle.figure("single")
    assert abs(fig.get_figwidth() - 3.5) < 1e-9
    ax.plot([0, 1], [0, 1], color=plotstyle.colour("tree"))
    ax.set_xlabel("Number of trees")
    path = plotstyle.save(fig, tmp_path / "out" / "fig.pdf")
    assert path.read_bytes().startswith(b"%PDF")
    fig, _ = plotstyle.figure("double")
    assert abs(fig.get_figwidth() - 7.16) < 1e-9
    assert plotstyle.colour("tree") != plotstyle.colour("knn")
