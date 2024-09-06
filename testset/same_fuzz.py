import difflib
from rapidfuzz import fuzz
import sys
# Load lines from a file into a list
import difflib
import hashlib
from concurrent.futures import ThreadPoolExecutor

# Load lines from a file into a list
def load_lines(file_path):
    with open(file_path, 'r') as file:
        return file.read().splitlines()

# Hash a line using SHA-256 for exact match lookups
def hash_line(line):
    return hashlib.sha256(line.encode('utf-8')).hexdigest()

# Function to check a single line against all lines in the first file
def check_line(line_src, line_src_tr, line2, lines_file1, hashed_file1, threshold, sub="nosub"):
    # Check if exact match exists using the hash
    #if hash_line(line2 in hashed_file1:
     #   return ('exact', line2)
    if sub=="sub":
        for line1 in lines_file1:
            if len(line2)>len(line1): continue
            if line2 in line1:
                return ('exact', (line2, line_src, line_src_tr))
        for line1 in lines_file1:
            if len(line2)>len(line1): continue
            if fuzz.partial_ratio(line2, line1) >= threshold:
                return ('similar', (line2, line1, line_src, line_src_tr))
    elif sub=="nosub":
        for line1 in lines_file1:
            if line2 == line1:
                return ('exact', (line2, line_src, line_src_tr))
        for line1 in lines_file1:
            if fuzz.ratio(line2, line1) >= threshold:
                return ('similar', (line2, line1, line_src, line_src_tr))
    elif sub=="both":
        for line1 in lines_file1:
            if line2 in line1:
                return ('exact', (line2, line_src, line_src_tr))
        for line1 in lines_file1:
            if fuzz.ratio(line2, line1) >= threshold:
                return ('similar', (line2, line1, line_src, line_src_tr))
            if len(line2)<len(line1) and fuzz.partial_ratio(line2, line1) >= threshold:
                return ('similar', (line2, line1, line_src, line_src_tr))


    print("done") 
    # If no match is found
    return ('unique', (line2, line_src, line_src_tr))

# Compare the lines from the two files
def compare_files(file_src,file_src_tr,file1, file2, threshold=0.7, sub="sub", prefix=""):
    # Load lines from the files
    lines_file_src = load_lines(file_src)
    lines_file_src_tr = load_lines(file_src_tr)
    lines_file1 = load_lines(file1)
    lines_file2 = load_lines(file2)

    # Create a set of hashed lines for exact match lookup
    hashed_file1 = {hash_line(line) for line in lines_file1}

    # Use ThreadPoolExecutor to parallelize the comparison
    exact_matches = []
    similar_lines = []

    unique_lines_file2 = []

    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = [executor.submit(check_line, line_src, line_src_tr, line2, lines_file1, hashed_file1, threshold, sub=sub) for line2,line_src,line_src_tr in zip(lines_file2,lines_file_src,lines_file_src_tr)]

        for future in futures:
            result_type, result = future.result()
            if result_type == 'exact':
                exact_matches.append(result)
            elif result_type == 'similar':
                similar_lines.append(result[0]+'\t'+result[1])
#            elif result_type == 'similar_sub':
 #               similar_sub_lines.append(result[0]+'\t'+result[1])
  #          elif result_type == 'exact_sub':
   #             exact_sub_matches.add(result)



            elif result_type == 'unique':
                unique_lines_file2.append(result[1]+'\t'+result[2]+'\t'+result[0])

            

    # Print the results
    print(f"Number of exact matches: {len(exact_matches)}")
    print(f"Number of similar lines: {len(similar_lines)}")
    print(f"exact+similar: {len(similar_lines)+len(exact_matches)}")

    #print(f"Number of exact sub matches: {len(exact_sub_matches)}")
#    print(f"Number of similar sub lines: {len(similar_sub_lines)}")
 #   print(f"exact sub+similar sub: {len(similar_sub_lines)+len(exact_sub_matches)}")


    print(f"Number of unique lines in the second file: {len(unique_lines_file2)}")

    # Optionally, print the actual similar lines (if needed)
    # print("Similar lines (from file2 -> file1):")
    # for line2, line1 in similar_lines:
    #     print(f"'{line2}' is similar to '{line1}'")

    # print("Unique lines in the second file:")
    with open(f"unique_fuzzy_{prefix}_{threshold}_{sub}.txt", 'w') as u:
        for line in unique_lines_file2:
            u.write(line)
            u.write('\n')
    with open(f"similar_fuzzy_{prefix}_{threshold}_{sub}.txt", 'w') as u:
        for line in similar_lines:
            u.write(line)
            u.write('\n')


# Define the file paths
file_src_path = sys.argv[1]
file_src_tr_path = sys.argv[2]

file1_path = sys.argv[3]
file2_path = sys.argv[4]


# Define the edit distance threshold
threshold = int(sys.argv[5])
sub=sys.argv[6]
prefix=sys.argv[7]
# Run the comparison
compare_files(file_src_path, file_src_tr_path, file1_path, file2_path, threshold, sub=sub,prefix=prefix)

