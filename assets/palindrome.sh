#!/bin/bash

# Read input from user
read -p "Enter a string: " str

# Reverse the string
rev_str=$(echo "$str" | rev)

# Check if original and reversed strings are the same
if [ "$str" == "$rev_str" ]; then
    echo "The string '$str' is a palindrome."
else
    echo "The string '$str' is NOT a palindrome."
fi

