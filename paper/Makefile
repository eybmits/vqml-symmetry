.PHONY: figures paper clean

figures: fig1_4panel_standalone.pdf
	python make_figures.py

# Figure 1 is the canonical four-panel standalone TikZ/PDF composite.
fig1_4panel_standalone.pdf: fig1_4panel_standalone.tex txt/fig_protocol_tikz_4panel.tex
	latexmk -pdf -interaction=nonstopmode fig1_4panel_standalone.tex

paper: figures
	latexmk -pdf -interaction=nonstopmode main.tex

clean:
	latexmk -C main.tex
	latexmk -C fig1_4panel_standalone.tex
