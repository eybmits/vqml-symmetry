Final seminar paper project.

Compile main.tex with PDFLaTeX. To update the word and character counts automatically, compile with shell escape enabled because the template uses texcount:

  pdflatex -shell-escape main.tex
  bibtex main
  pdflatex -shell-escape main.tex
  pdflatex -shell-escape main.tex

The final compiled PDF is included as main.pdf.
