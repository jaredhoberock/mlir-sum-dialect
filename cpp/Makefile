ifndef MLIR_SYS_220_PREFIX
$(error MLIR_SYS_220_PREFIX must be set; point it at your LLVM/MLIR 22 install)
endif

BUILD_DIR := build
LLVM_BIN_PATH := $(MLIR_SYS_220_PREFIX)/bin
LLVM_CONFIG := $(LLVM_BIN_PATH)/llvm-config
TBLGEN := $(LLVM_BIN_PATH)/mlir-tblgen

CXX := clang++
CXXFLAGS := -Wall -Werror -fPIC $(shell $(LLVM_CONFIG) --cxxflags)
INCLUDES := -I$(BUILD_DIR) -I. -I$(MLIR_SYS_220_PREFIX)/include

DIALECT_SOURCES := c_api.cpp ConvertToLLVM.cpp ConvertToSCF.cpp Sum.cpp SumOps.cpp SumTypeInterface.cpp SumTypes.cpp
DIALECT_OBJECTS := $(addprefix $(BUILD_DIR)/,$(DIALECT_SOURCES:.cpp=.o))
PLUGIN_OBJECTS := $(DIALECT_OBJECTS) $(BUILD_DIR)/Plugin.o
TD_FILES := $(wildcard *.td)

GENERATED := \
  $(BUILD_DIR)/Sum.hpp.inc \
  $(BUILD_DIR)/Sum.cpp.inc \
  $(BUILD_DIR)/SumOps.hpp.inc \
  $(BUILD_DIR)/SumOps.cpp.inc \
  $(BUILD_DIR)/SumTypeInterface.hpp.inc \
  $(BUILD_DIR)/SumTypeInterface.cpp.inc \
  $(BUILD_DIR)/SumTypes.hpp.inc \
  $(BUILD_DIR)/SumTypes.cpp.inc

.PHONY: all clean test

all: $(BUILD_DIR)/libsum_dialect.a $(BUILD_DIR)/libsum_dialect.so

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/Sum.hpp.inc: Sum.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-dialect-decls $(INCLUDES) $< -o $@

$(BUILD_DIR)/Sum.cpp.inc: Sum.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-dialect-defs $(INCLUDES) $< -o $@

$(BUILD_DIR)/SumOps.hpp.inc: SumOps.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-op-decls $(INCLUDES) $< -o $@

$(BUILD_DIR)/SumOps.cpp.inc: SumOps.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-op-defs $(INCLUDES) $< -o $@

$(BUILD_DIR)/SumTypeInterface.hpp.inc: SumTypeInterface.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-type-interface-decls $(INCLUDES) $< -o $@

$(BUILD_DIR)/SumTypeInterface.cpp.inc: SumTypeInterface.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-type-interface-defs $(INCLUDES) $< -o $@

$(BUILD_DIR)/SumTypes.hpp.inc: SumTypes.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-typedef-decls $(INCLUDES) $< -o $@

$(BUILD_DIR)/SumTypes.cpp.inc: SumTypes.td $(TD_FILES) | $(BUILD_DIR)
	$(TBLGEN) --gen-typedef-defs $(INCLUDES) $< -o $@

$(BUILD_DIR)/%.o: %.cpp $(GENERATED) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/libsum_dialect.a: $(DIALECT_OBJECTS)
	ar rcs $@ $^

$(BUILD_DIR)/libsum_dialect.so: $(PLUGIN_OBJECTS)
	$(CXX) -shared $^ -o $@

test: $(BUILD_DIR)/libsum_dialect.so
	SUM_DIALECT_PLUGIN=$(CURDIR)/$(BUILD_DIR)/libsum_dialect.so lit tests

clean:
	rm -rf $(BUILD_DIR)
