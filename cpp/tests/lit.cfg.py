import os
import lit.formats

config.name = "Sum Dialect Tests"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir']
config.test_source_root = os.path.dirname(__file__)

# Every tool and plugin path comes from the site configuration build.rs writes
# into the build's output directory (lit.site.cfg.py), which loads this file.
def tool(name):
    return os.path.join(config.llvm_tools_dir, name)

def utility(name):
    return os.path.join(config.llvm_utils_dir, name)

# The directories `tool` and `utility` look in are also on the path a row's shell
# searches, so a row inverting an exit status writes `not` and gets LLVM's. It is
# a path entry rather than a substitution: a substitution rewrites every
# occurrence of its own name in the whole line, and `not` is a word a RUN line may
# hold for other reasons.
config.environment['PATH'] = os.pathsep.join(
    [config.llvm_tools_dir, config.llvm_utils_dir, config.environment['PATH']])
# The dialect plugins in the order the site configuration records them: each
# after the plugins whose symbols it references, which must be loaded first.
dialect_plugins = ' '.join(f'--load-dialect-plugin={plugin}' for plugin in config.dialect_plugins)
# The lowering driver's pass plugin registers `leaf-conversion`, so a row runs the
# sum's leaf contribution at every boundary the conversion flattens. Its passes
# reach the registry while the command line is read, so a row names them in a
# -pass-pipeline.
config.substitutions.append(('mlir-opt', f'{tool("mlir-opt")} {dialect_plugins} --load-pass-plugin={config.lowering_driver_plugin}'))
config.substitutions.append(('mlir-runner', tool('mlir-runner')))
config.substitutions.append(('%mlir_c_runner_utils', os.path.join(config.llvm_lib_dir, 'libmlir_c_runner_utils.so')))
config.substitutions.append(('FileCheck', utility('FileCheck')))
