#!/bin/sh
echo "extern const unsigned char $2[] = {" > $1
cat $3 | xxd -i >> $1
echo "};" >> $1
echo "extern const unsigned $2_size = sizeof($2);" >> $1
