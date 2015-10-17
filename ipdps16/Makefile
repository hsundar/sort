.PHONY: all clean

SRC = paper.tex
TARGET = paper.pdf

all: $(TARGET)

$(TARGET): $(SRC)
	pdflatex $<
	bibtex $<
	pdflatex $<
	pdflatex $<

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.pdf $(TARGET)
