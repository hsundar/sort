AR=ar
CXX=mpicxx
RM=rm -r -f
MKDIRS=mkdir -p
INCDIR=-I./include -I./include/par -I./include/binOps -I./include/seq -I./include/omp_par
SRCDIR=./src
OBJDIR=./obj
LIBDIR=./lib
BINDIR=./bin

CXXFLAGS= -O2 -fopenmp $(INCDIR) #-D USE_OLD_SORT
LFLAGS= -r
LIBS=


CCFILES=main.cpp
CFILES=par/parUtils.C binOps/binUtils.C

OBJS=$(CCFILES:%.cpp=$(OBJDIR)/%.o) \
     $(CFILES:%.C=$(OBJDIR)/%.o)

TARGET=$(BINDIR)/main

all : $(TARGET) 


clean :
	$(RM) $(OBJDIR)/* $(BINDIR)/* $(LIBDIR)/*

$(TARGET) : $(OBJS)
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@

$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX) -c $(CXXFLAGS) $^ -o $@

$(OBJDIR)/%.o : $(SRCDIR)/%.C
	-@$(MKDIRS) $(dir $@)
	$(CXX) -c $(CXXFLAGS) $^ -o $@
