# remove partially created targets
.DELETE_ON_ERROR:
# clean and all are not real files
.PHONY: all clean

#
# Definitions
#

LATEX = pdflatex -halt-on-error
#LATEX += -interaction=batchmode
BIBTEX = bibtex
RM_F = rm -f
PDF_VIEWER = evince

CLS_FORMAT =
BST_FORMAT =

PAPER = report

#
# Sources for paper
#

SRCS = \
	$(PAPER).tex					\
	Introduction.tex					\
	RelatedWork.tex				\
    Dataset.tex                     \
    DecisionTree.tex                    \
	Methods.tex					\
	Results.tex				\
    Discussion.tex                      \
	Conclusion.tex					\

#
# Images files used in paper
#
#IMAGES = \
	SWAMP-dev-integration.pdf \



#
# Generated files (removed by clean target)
#

CLEAN_FILES = \
	$(PAPER).aux					\
	$(PAPER).bbl					\
	$(PAPER).blg					\
	$(PAPER).log					\
	$(PAPER).pdf					\
	$(PAPER).toc					\
	$(PAPER).out


#
# Rules
#

PAPER_PDF = $(PAPER).pdf
all:	$(PAPER_PDF)

$(PAPER_PDF): $(PAPER).bbl $(SRCS) $(IMAGES) $(CLS_FORMAT)
	$(LATEX) $(PAPER)	\
	&& echo -e  \\n\\n\\n\\nFinal pass\\n	\
	&& $(LATEX) $(PAPER)

$(PAPER).bbl: references.bib $(SRCS) $(BST_FORMAT)
	rm -f $@ 		\
	&& $(LATEX) $(PAPER)	\
	&& $(BIBTEX) $(PAPER)


clean:
	$(RM_F) $(CLEAN_FILES)

view: $(PAPER_PDF)
	$(PDF_VIEWER) $(PAPER_PDF)
