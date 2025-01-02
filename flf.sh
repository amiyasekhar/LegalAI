while read -r line; do
    SHA=$(echo "$line" | cut -d' ' -f1)
    FILE=$(echo "$line" | cut -d' ' -f2-)
    SIZE=$(git cat-file -s "$SHA")
    # e.g. check for > 10 MB:
    if [ $SIZE -gt 10485760 ]; then
        echo "$((SIZE/1024/1024)) MB - $FILE ($SHA)"
    fi
done < allfiles.txt