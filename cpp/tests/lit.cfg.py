import os
import lit.formats

config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)

mlir_prefix = os.environ.get('MLIR_SYS_230_PREFIX', '/home/jhoberock/dev/git/llvm-project-23/install-release-asserts-23.1.4')
llvm_bin = os.path.join(mlir_prefix, 'bin')
llvm_lib = os.path.join(mlir_prefix, 'lib')
fallback_llvm_bin = '/home/jhoberock/dev/git/llvm-project-23/build-release-asserts/bin'

def tool(name):
    installed = os.path.join(llvm_bin, name)
    if os.path.exists(installed):
        return installed
    return os.path.join(fallback_llvm_bin, name)

def plugin(env_var, fallback):
    return os.environ.get(env_var, fallback)

# Tools the RUN lines name bare (`not`) resolve the same way `tool()` does: the
# install's bin first, then the build tree's.
config.environment['PATH'] = os.pathsep.join([llvm_bin, fallback_llvm_bin, os.environ.get('PATH', '')])

config.name = "Sum Dialect Tests"
sum_plugin = plugin('SUM_DIALECT_PLUGIN', os.path.join(os.path.dirname(__file__), '..', 'build', 'libsum_dialect.so'))
# The lowering driver's pass plugin registers `leaf-conversion`, so a row runs the
# sum's leaf contribution at every boundary the conversion flattens. Its passes
# reach the registry while the command line is read, so a row names them in a
# -pass-pipeline.
driver_plugin = plugin('LOWERING_DRIVER_PLUGIN', os.path.join(os.path.dirname(__file__), '../../../mlir-lowering-driver/cpp/build/liblowering_driver.so'))
config.substitutions.append(('mlir-opt', f'{tool('mlir-opt')} --load-dialect-plugin={sum_plugin} --load-pass-plugin={driver_plugin}'))
config.substitutions.append(('mlir-runner', tool('mlir-runner')))
config.substitutions.append(('%mlir_c_runner_utils', os.path.join(llvm_lib, 'libmlir_c_runner_utils.so')))
config.substitutions.append(('FileCheck', tool('FileCheck')))
