.PHONY: figures paper clean

figures:
	python make_figures.py

paper: figures
	latexmk -pdf -interaction=nonstopmode main.tex

clean:
	latexmk -C main.tex
