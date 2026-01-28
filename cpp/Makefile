LLVM_BIN_PATH = $(HOME)/dev/git/llvm-project-20/install/bin

LLVM_CONFIG := $(LLVM_BIN_PATH)/llvm-config
TBLGEN := $(LLVM_BIN_PATH)/mlir-tblgen
OPT := $(LLVM_BIN_PATH)/mlir-opt

# Compiler flags
CXX := clang++
CXXFLAGS := -Wall -Werror -g -fPIC `$(LLVM_CONFIG) --cxxflags`

# LLVM/MLIR libraries
MLIR_INCLUDE = /home/jhoberock/dev/git/llvm-project-20/install/include

INCLUDES := -I $(MLIR_INCLUDE)

# Dialect library sources
DIALECT_SOURCES := c_api.cpp ConvertToLLVM.cpp ConvertToSCF.cpp Sum.cpp SumOps.cpp SumTypes.cpp
DIALECT_OBJECTS := $(DIALECT_SOURCES:.cpp=.o)

# Generated files
GENERATED := Sum.hpp.inc Sum.cpp.inc SumOps.hpp.inc SumOps.cpp.inc SumTypes.hpp.inc SumTypes.cpp.inc

.PHONY: all clean

all: libsum_dialect.a libsum_dialect.so

# TableGen rules
Sum.hpp.inc: Sum.td
	$(TBLGEN) --gen-dialect-decls $(INCLUDES) $< -o $@

Sum.cpp.inc: Sum.td
	$(TBLGEN) --gen-dialect-defs $(INCLUDES) $< -o $@

SumOps.hpp.inc: SumOps.td
	$(TBLGEN) --gen-op-decls $(INCLUDES) $< -o $@

SumOps.cpp.inc: SumOps.td
	$(TBLGEN) --gen-op-defs $(INCLUDES) $< -o $@

SumTypes.hpp.inc: SumTypes.td
	$(TBLGEN) --gen-typedef-decls $(INCLUDES) $< -o $@

SumTypes.cpp.inc: SumTypes.td
	$(TBLGEN) --gen-typedef-defs $(INCLUDES) $< -o $@

# Object file rules
%.o: %.cpp $(GENERATED)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

libsum_dialect.a: $(DIALECT_OBJECTS)
	ar rcs $@ $^

PLUGIN_OBJECTS := $(DIALECT_OBJECTS) Plugin.o

libsum_dialect.so: $(PLUGIN_OBJECTS)
	$(CXX) -shared $^ -o $@

.PHONY: test
test: libsum_dialect.so
	@echo "Running sum dialect tests..."
	lit tests

clean:
	rm -f *.o *.inc *.a *.so
