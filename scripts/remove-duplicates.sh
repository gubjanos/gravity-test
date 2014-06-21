#!/bin/bash
# This script removes duplicate entries from the file
# given as first parameter, and prints it to the file
# givan as second parameter.
sort $1 | awk 'NR%2==1' | shuf > $2
