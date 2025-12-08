#pragma once

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;

namespace sum {

void populateSumToLLVMConversionPatterns(LLVMTypeConverter& typeConverter,
                                           RewritePatternSet& patterns);
}
}
