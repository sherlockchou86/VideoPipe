#!/bin/bash

####################################
# run release.sh 
# to collect libraries & header files used out of VideoPipe workspace.
####################################
SO_SRC_DIRS=(
    "../build/libs"
)
H_SRC_DIRS=(
    "../excepts"
    "../nodes"
    "../objects"
    "../third_party"
    "../utils"
)

LIBS_DIR="./libs"
INCLUDE_DIR="./include"
CHECK_ELF_SO=true

echo "🧹 removing existing files..."
rm -rf "$LIBS_DIR" "$INCLUDE_DIR"

mkdir -p "$LIBS_DIR"
mkdir -p "$INCLUDE_DIR"

echo "✅ collecting libraries & header files..."
echo "target library path: $LIBS_DIR"
echo "target header path: $INCLUDE_DIR"
echo

collect_so_files() {
    local src_dir
    for src_dir in "${SO_SRC_DIRS[@]}"; do
        if [ ! -d "$src_dir" ]; then
            echo "⚠️ warn: .so source folder not exists, ignore: $src_dir"
            continue
        fi

        echo "🔍 scanning .so files: $src_dir"
        find "$src_dir" -type f -name "*.so*" | while read -r so_file; do
            if $CHECK_ELF_SO; then
                if ! file "$so_file" 2>/dev/null | grep -q "ELF.*shared object"; then
                    continue
                fi
            fi

            rel_path="${so_file#$src_dir/}"
            dest_path="$LIBS_DIR/$rel_path"
            dest_dir=$(dirname "$dest_path")

            mkdir -p "$dest_dir"
            cp "$so_file" "$dest_path"
            echo "📦 copied .so: $so_file -> $dest_path"
        done
    done
}

collect_h_files() {
    local src_dir
    for src_dir in "${H_SRC_DIRS[@]}"; do
        if [ ! -d "$src_dir" ]; then
            echo "⚠️ warn: .h source folder not exists, ignore: $src_dir"
            continue
        fi

        dir_name=$(basename "$src_dir")

        echo "🔍 scanning .h files: $src_dir (keep directory name: $dir_name)"

        find "$src_dir" -type f \( -name "*.h" -o -name "*.hpp" -o -name "*.hxx" \) | while read -r h_file; do
            rel_path="${h_file#$src_dir/}"

            dest_path="$INCLUDE_DIR/$dir_name/$rel_path"
            dest_dir=$(dirname "$dest_path")

            mkdir -p "$dest_dir"
            cp "$h_file" "$dest_path"
            echo "📘 copied .h: $h_file -> $dest_path"
        done
    done
}

collect_so_files
echo
collect_h_files

echo
echo "🎉 completely！"
echo "libraries saved to: $LIBS_DIR"
echo "headers saved to: $INCLUDE_DIR"