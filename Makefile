AR=ar
CXX=mpicxx
RM=rm -r -f
MKDIRS=mkdir -p
INCDIR=-I./include -I./include/oct -I./include/point -I./include/par -I./include/binOps -I./include/seq -I./include/omp_par -I./include/sse -I./include/gensort -I./include/avx -I$(PAPI_INC)
SRCDIR=./src
OBJDIR=./obj
LIBDIR=./lib
BINDIR=./bin
TMPDIR=./tmp

KWAY=2
CXXFLAGS= -O3 -fopenmp $(INCDIR) -D__USE_64_BIT_INT__ -DALLTOALLV_FIX -D_PROFILE_SORT -DKWAY=$(KWAY) # -D__VERIFY__# -D__DEBUG_PAR__ # -DKWICK -D USE_OLD_SORT
LFLAGS= -r
LIBS=-L$(PAPI_LIB) -lpapi


CCFILES=main.cpp
CFILES=par/parUtils.C par/sort_profiler.C binOps/binUtils.C oct/TreeNode.C gensort/gensort.C gensort/rand16.C

OBJS=$(CCFILES:%.cpp=$(OBJDIR)/%.o) \
     $(CFILES:%.C=$(OBJDIR)/%.o)

TARGET=$(BINDIR)/main

all : $(TARGET) 

hyper : CXXFLAGS += -DKWICK
hyper : $(OBJS)
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@$(KWAY)

sample : $(OBJS)
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@$(KWAY)

swapsort : CXXFLAGS += -DSWAPRANKS -DKWICK
swapsort: $(OBJS)
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@ 

clean :
	$(RM) $(OBJDIR)/* $(BINDIR)/* $(LIBDIR)/* $(TMPDIR)/*

$(TARGET) : $(OBJS)
	-@$(MKDIRS) $(dir $@)
	$(CXX) $(CXXFLAGS) $(LIBS) $^ -o $@

$(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	-@$(MKDIRS) $(dir $@)
	$(CXX) -c $(CXXFLAGS) $^ -o $@

$(OBJDIR)/%.o : $(SRCDIR)/%.C
	-@$(MKDIRS) $(dir $@)
	$(CXX) -c $(CXXFLAGS) $^ -o $@
