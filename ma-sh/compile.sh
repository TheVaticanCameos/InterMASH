if [ "$(uname)" = "Darwin" ]; then
  PROCESSOR_NUM=$(sysctl -n hw.physicalcpu)
elif [ "$(uname)" = "Linux" ]; then
  PROCESSOR_NUM=$(cat /proc/cpuinfo | grep "processor" | wc -l)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

export MAX_JOBS=${PROCESSOR_NUM}

export CC=$(which gcc)
export CXX=$(which g++)
echo "Using CC: $CC"
echo "Using CXX: $CXX"

if [ -f /usr/local/cuda/include/cuda_runtime.h ]; then
  export CUDA_HOME=/usr/local/cuda
  export PATH="$CUDA_HOME/bin:$PATH"
fi

export PIP_NO_BUILD_ISOLATION=1

python -m pip uninstall mash-cpp -y

rm -rf ../ma-sh/build
rm -rf ../ma-sh/*.egg-info
rm -f ../ma-sh/*.so

# bear -- python setup.py build_ext --inplace
python setup.py build_ext --inplace
if [ -f compile_commands.json ]; then
  mkdir -p build
  mv compile_commands.json build
fi

python -m pip install . --no-build-isolation --no-deps
