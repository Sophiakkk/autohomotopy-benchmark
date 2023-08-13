#!/bin/bash

# Declare a string array with type
declare -a StringArray=("Windows XP" "Windows 10" 
  "Windows ME" "Windows 8.1" "Windows Server 2016" )

# Read the array values with space
for val in "${StringArray[@]}"; do
 echo $val
done