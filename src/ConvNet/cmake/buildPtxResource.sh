#!/bin/sh
echo "extern const char $2[] = \"\\" > $1
while read line
do
    echo "${line//\"/\\\"}\n\\" >> $1
done < $3
echo "\";" >> $1
echo "extern const unsigned $2_size = sizeof($2);" >> $1
