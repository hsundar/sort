.PHONY: all clean

SRC = sc15-sort.tex
TARGET = sc15-sort.pdf

all: $(TARGET)

$(TARGET): $(SRC)
	pdflatex $<
	bibtex $<
	pdflatex $<
	pdflatex $<

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.pdf $(TARGET)
