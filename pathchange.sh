#!/bin/bash

# Get the current directory
current_dir=$(pwd)
current_dir_data=$(pwd)/Data
current_dir_mat=$(pwd)/Data/CD_data_share
# Recursively find all .py files in the current directory and its subdirectories
find "$current_dir" -type f -name "*.py" | while read file; do
    # Count how many occurrences of "/mnt/HD2/czr/32ant*32car_3.5GHz_40MHz_R501-1400_V1_23.9.25/mat" are in the file before replacing

    replacements_in_file=$(grep -o "/mnt/HD2/czr/32ant\*32car_3.5GHz_40MHz_R501-1400_V1_23.9.25/mat" "$file" | wc -l)

    # If there are any replacements to make
    if [ "$replacements_in_file" -gt 0 ]; then
        # Perform the replacement in the file
        sed -i "s|/mnt/HD2/czr/32ant\*32car_3.5GHz_40MHz_R501-1400_V1_23.9.25/mat|$current_dir_mat|g" "$file"

        # Output which file was updated and how many occurrences were replaced
        echo "Updated file: $file (replaced $replacements_in_file occurrences)"
    fi



    replacements_in_file=$(grep -o "/mnt/HD2/czr/32ant\*32car_3.5GHz_40MHz_R501-1400_V1_23.9.25" "$file" | wc -l)
    # If there are any replacements to make
    if [ "$replacements_in_file" -gt 0 ]; then
        # Perform the replacement in the file
        sed -i "s|/mnt/HD2/czr/32ant\*32car_3.5GHz_40MHz_R501-1400_V1_23.9.25|$current_dir_data|g" "$file"

        # Output which file was updated and how many occurrences were replaced
        echo "Updated file: $file (replaced $replacements_in_file occurrences)"
    fi

done

# Output the total number of replacements
echo "Replacement complete! "
