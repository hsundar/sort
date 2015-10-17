.PHONY: all clean

SRC = paper.tex
TARGET = paper.pdf

all: $(TARGET)

$(TARGET): $(SRC)
	pdflatex -shell-escape $<
	bibtex $<
	pdflatex -shell-escape $<
	pdflatex -shell-escape $<

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.pdf $(TARGET)
